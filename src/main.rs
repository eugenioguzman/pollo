#![feature(isqrt)]
#![feature(iterator_try_collect)]

use std::{borrow::Cow, cmp::max, collections::VecDeque, fs::File, io::Write, iter::zip, path::PathBuf};
use clap::{Parser, Subcommand, ValueEnum};
use iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelBridge, ParallelIterator};
use ndarray::{azip, Array1, Array2, Axis};
use rust_htslib::bcf::{self, Read};
use serde::{Deserialize, Serialize};
use rayon::*;
use triangular_matrix::{triangular_matrix_ij, triangular_matrix_len, TriangularMatrix};
mod triangular_matrix;

const PLOIDY: usize = 2;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(short, long, default_value_t=1, value_parser = clap::value_parser!(u16).range(1..))]
    threads: u16,
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Copy, Clone, ValueEnum)]
enum VcfOutputType {
    V, Z, U, B
}

#[derive(Subcommand)]
enum Commands {
    GenerateTest { file: PathBuf },
    Run {
        input_file: PathBuf,
        #[arg(short='o', long)]
        output: Option<PathBuf>,
        #[arg(short, long="stats")]
        stats_file: Option<PathBuf>,
        #[arg(short='g', long)]
        max_guiders: Option<usize>,
        #[arg(value_enum, short='O', long)]
        output_type: Option<VcfOutputType>
    },
}

fn make_test(file: &PathBuf) -> Result<(), rust_htslib::tpool::Error> {
    let samps = [
        [0,1],
        [1,2],
        [0,2],
        [1,1]
    ];
    let chr_vers: [[u8; 10]; 3] = [
        [1,0,0,0,1,0,0,1,0,1],
        [0,1,1,0,1,0,1,0,1,0],
        [1,0,1,0,1,1,0,1,1,1]
    ];

    let mut header = bcf::Header::new();
    let base_letters =  "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    for [a, b] in samps {
        let id = 
            String::from(base_letters.chars().nth(a).unwrap_or('_')) +
            &String::from(base_letters.chars().nth(b).unwrap_or('_'));
        header.push_sample(id.as_bytes());

    }   
    header.push_record(format!(
        "##contig=<ID={},length={}>",
        "Z", 100).as_bytes());
    header.push_record(format!(
        r#"##FORMAT=<ID={},Number={},Type={},Description="{}">"#,
        "GT", 1, "String", "Genotype").as_bytes());
    let mut writer = bcf::Writer::from_path(file, &header, true, bcf::Format::Vcf)?;


    for i in 0..chr_vers[0].len() {
        let mut record = writer.empty_record();
        record.set_alleles(&[b"G", b"T"])?;
        record.set_rid(record.header().name2rid(b"Z").ok());
        record.set_pos(i.try_into().unwrap_or(0));
        let mut gts = vec![];
        for [a, b] in samps {
            let mut subgts: Vec<i32> = Vec::with_capacity(2);
            subgts.push(chr_vers[a][i].into());
            subgts.push(chr_vers[b][i].into());
            subgts.sort();
            gts.extend(subgts.iter().map(|x| bcf::record::GenotypeAllele::Unphased(*x)));
        }
        record.push_genotypes(&gts)?;
        writer.write(&record)?;
    }
    Ok(())
}


type Allele = Option<i32>;

fn get_allele(ga: &bcf::record::GenotypeAllele) -> Allele {
    match ga {
        bcf::record::GenotypeAllele::Unphased(x) => {
            Some(*x)
        },
        bcf::record::GenotypeAllele::Phased(x) => {
            Some(*x)
        },
        bcf::record::GenotypeAllele::UnphasedMissing => None,
        bcf::record::GenotypeAllele::PhasedMissing => None,
    }
}

#[derive(thiserror::Error, Debug)]
enum QuickReadError {
    #[error("htslib error")]
    BcfError(#[from] rust_htslib::tpool::Error),
    #[error("integer conversion error")]
    UsizeConvError(#[from] std::num::TryFromIntError)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Dists {
    dist: TriangularMatrix<u128>,
    cmps: TriangularMatrix<u128>
}

fn quick_read(file: &PathBuf) -> Result<(Vec<Option<Dists>>, Vec<String>, Vec<String>), QuickReadError> {
    let mut reader = bcf::Reader::from_path(file)?;
    let samples: Vec<String> = bcf::Read::header(&reader).samples().iter().map(|v| String::from_utf8_lossy(*v).into_owned()).collect();
    let tri_iter = Vec::from_iter((0..triangular_matrix_len(samples.len())).map(|i| triangular_matrix_ij(i)));
    let mut chrom_dists = vec![None; reader.header().contig_count() as usize];
    for (rn, rec_result) in reader.records().enumerate() {
        let rec = match rec_result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping record {} due to problem: {}", rn, e.to_string());
                continue
            }
        };

        let rid = match rec.rid() {
            Some(x) => x,
            None => continue
        };


        let cd = match chrom_dists.get_mut(rid as usize) {
            Some(x) => match x {
                Some(y) => y,
                None => {
                    let y = Dists {
                        dist: TriangularMatrix::zeros(samples.len()),
                        cmps: TriangularMatrix::zeros(samples.len())
                    };
                    *x = Some(y);
                    x.as_mut().unwrap()
                },
            },
            None => continue,
        };
        
        let gts_raw = rec.genotypes()?;
        let mut poisoned = vec![false; samples.len()];
        let mut rows = Array2::<i8>::zeros([samples.len(), rec.allele_count() as usize]);
        for i in 0..samples.len() {
            let gt = gts_raw.get(i);
            for a in gt.iter() {
                match get_allele(a) {
                    Some(k) => { rows[[i, k as usize]] += 1; },
                    None => { poisoned[i] = true; },
                }
            };
        }
        
        vec![vec![0; rec.allele_count().try_into()?]; samples.len()];

        zip(tri_iter.iter(), zip(cd.dist.arr.iter_mut(), cd.cmps.arr.iter_mut())).par_bridge().for_each(|((i, j), (dist, cmps))| {
            let p1 = poisoned[*i];
            let p2 = poisoned[*j];
            if p1 || p2 {
                return
            }
            
            let distance = if j != i { 
                let mut diff = Array1::zeros([rec.allele_count() as usize]);
                azip!((gt1 in rows.row(*i), gt2 in rows.row(*j), d in &mut diff) *d = (gt1 - gt2).max(0));
                diff.sum()
            } else { 0 };
            *dist += distance as u128;
            *cmps += 1;
        });
    }
    let mut chrom_names = Vec::with_capacity(chrom_dists.len());
    for i in 0..chrom_dists.len() {
        chrom_names.push(String::from_utf8_lossy(reader.header().rid2name(i.try_into()?).unwrap()).into_owned());
    }
    Ok((chrom_dists, chrom_names, samples))
}


struct GuiderGenerator {
    i_viewer: Vec<usize>,
    j_viewer: Vec<usize>
}

impl GuiderGenerator {
    fn new(nsamples: usize) -> GuiderGenerator {
        let (iv, jv) = (0..triangular_matrix_len(nsamples)).map(|i| triangular_matrix_ij(i)).unzip();
        GuiderGenerator {
            i_viewer: iv,
            j_viewer: jv
        }
    }
    fn get_ij_arrs<T: Clone>(&self, row: &Array1<T>) -> (Array1<T>, Array1<T>) {
        let i = row.select(Axis(0), &self.i_viewer);
        let j = row.select(Axis(0), &self.j_viewer);
        (i, j)
    }
}

fn deque_insert_replace<T>(deque: &mut VecDeque<T>, value: T, lim: usize) {
    if deque.len() < lim {
        deque.push_front(value);
    } else {
        deque.pop_back();
        deque.push_front(value);
    }
}

type Guider = [Vec<(usize, f64)>; PLOIDY];

fn gen_guiders(dists: &Vec<Option<Dists>>, limit: Option<usize>) -> Vec<Option<Vec<Guider>>> {
    let mut guiders = Vec::with_capacity(dists.len());
    for dist in dists {
        let d = match dist {
            Some(d) => d.clone(),
            None => {
                guiders.push(None);
                continue;
            },
        };
        let mut corr = -d.dist.map(|x| *x as f64) / (d.cmps.map(|x| *x as f64) * (PLOIDY as f64)) + 1f64;
        corr.fillna(0.);
        let corr = corr;
        let gg = GuiderGenerator::new(corr.size);
        let mut chr_guiders: Vec<Option<[Vec<(usize, f64)>; 2]>> = vec![None; corr.size];


        chr_guiders.par_iter_mut().enumerate().for_each(|(x, target)| {
            let arr_x = corr.row(x);
            let (arr_xi, arr_xj) = gg.get_ij_arrs(&arr_x);
            let arr_ij = &corr.arr;
            let result = TriangularMatrix {
                arr: arr_xi*arr_xj/arr_ij,
                //arr_ij - arr_xi,
                size: corr.size,
            };
            let ((guider_idx_i, guider_idx_j), _) = result.arr.iter().enumerate().par_bridge()
                .map(|(i, v)| (triangular_matrix_ij(i), *v))
                .filter(|((pi, pj), _)| (*pi != x) && (*pj != x) && (*pi != *pj))
                .reduce(
                    || ((0usize, 10usize), f64::NEG_INFINITY),
                    |((opi, opj), ov), ((npi, npj), nv)| {
                        if nv > ov {
                            ((npi, npj), nv)
                        } else {
                            ((opi, opj), ov)
                        }
                    }
                );
            
            let arr_i = corr.row(guider_idx_i);
            let arr_j = corr.row(guider_idx_j);
            let score = (arr_i - arr_j) * arr_x;
            let score = score.into_iter().enumerate();
            let mut guiders_i;
            let mut guiders_j;
            match limit {
                Some(lim) => {
                    let mut guiders_deque_i = VecDeque::with_capacity(lim);
                    let mut guiders_deque_j = VecDeque::with_capacity(lim);
                    let mut sum_i = 0.0;
                    let mut sum_j = 0.0;
                    for (s, v) in score.into_iter() {
                        if s == x {
                            continue;
                        }
                        if v > 0. {
                            let e = v.exp();
                            deque_insert_replace(&mut guiders_deque_i, (s, e), lim);
                            sum_i += e;
                        } else if v < 0. {
                            let e = v.exp();
                            deque_insert_replace(&mut guiders_deque_j, (s, e), lim);
                            sum_j += e;
                        }
                    }

                    guiders_i = Vec::from_iter(guiders_deque_i.into_iter().map(|(s, v)| (s, v/sum_i)));
                    guiders_j = Vec::from_iter(guiders_deque_j.into_iter().map(|(s, v)| (s, v/sum_j)));
                },
                None => {
                    guiders_i = Vec::with_capacity(corr.size);
                    guiders_j = Vec::with_capacity(corr.size);
                    let mut sum_i = 0.0;
                    let mut sum_j = 0.0;

                    for (s, v) in score {
                        if s == x {
                            continue;
                        }
                        if v > 0. {
                            let e = v.exp();
                            guiders_i.push((s, e));
                            sum_i += e;
                        } else if v < 0. {
                            let e = v.exp();
                            guiders_j.push((s, v.exp()));
                            sum_j += e;
                        }
                    }

                    for (_, v) in guiders_i.iter_mut() { *v /= sum_i; }
                    for (_, v) in guiders_j.iter_mut() { *v /= sum_j; }
                },
            }
            
            *target = Some([guiders_i, guiders_j]);
        });
        guiders.push(Some(Vec::from_iter(chr_guiders.into_iter().map(|v| v.unwrap()))));
    }
    guiders
}

fn to_genotype(alleles: &Vec<Allele>, phased: bool) -> Vec<bcf::record::GenotypeAllele> {
    let mut ordered_alleles = Cow::from(alleles);
    if !phased {
        ordered_alleles.to_mut().sort();
    }
    assert!(ordered_alleles.len() > 0);

    let mut genotype = Vec::with_capacity(ordered_alleles.len());
    
    genotype.push(match ordered_alleles[0] {
        Some(i) => bcf::record::GenotypeAllele::Unphased(i),
        None => bcf::record::GenotypeAllele::UnphasedMissing,
    });

    for al in ordered_alleles[1..].iter() {
        genotype.push(match al {
            Some(i) => if phased { bcf::record::GenotypeAllele::Phased(*i) } else { bcf::record::GenotypeAllele::Unphased(*i) },
            None => if phased { bcf::record::GenotypeAllele::PhasedMissing } else { bcf::record::GenotypeAllele::UnphasedMissing },
        });
    }
    genotype
}

fn copy_record_stuff(rec: &bcf::Record, new_rec: &mut bcf::Record) {
    new_rec.set_rid(rec.rid());
    new_rec.set_pos(rec.pos());
    new_rec.set_id(&rec.id()).unwrap();
    new_rec.set_alleles(&rec.alleles()).unwrap();
    new_rec.set_qual(rec.qual());
    let filt: Vec<bcf::header::Id> = rec.filters().into_iter().collect();
    let filt: Vec<&bcf::header::Id> = filt.iter().collect();
    new_rec.set_filters(&filt).unwrap();
}

fn write_phased(input_file: &PathBuf, output_file: &Option<PathBuf>, output_type: &VcfOutputType, guiders: &Vec<Option<Vec<Guider>>>) -> Result<Vec<Option<Dists>>, rust_htslib::tpool::Error> {
    let mut reader = bcf::Reader::from_path(input_file)?;
    let samples: Vec<String> = reader.header().samples().iter().map(|v| String::from_utf8_lossy(*v).into_owned()).collect();
    let mut header = bcf::Header::from_template(reader.header());
    header.push_record(format!(
        r#"##FORMAT=<ID={},Number={},Type={},Description="{}">"#,
        "PQ", 1, "Integer", "Phasing quality").as_bytes());

    let uncompressed = match output_type {
        VcfOutputType::V => true,
        VcfOutputType::U => true,
        _ => false
    };
    let htslib_format = match output_type {
        VcfOutputType::V => bcf::Format::Vcf,
        VcfOutputType::Z => bcf::Format::Vcf,
        VcfOutputType::U => bcf::Format::Bcf,
        VcfOutputType::B => bcf::Format::Bcf
    };
    let mut writer = match output_file {
        Some(f) => bcf::Writer::from_path(f, &header, uncompressed, htslib_format),
        None => bcf::Writer::from_stdout(&header, uncompressed, htslib_format)
    }?;
    let mut chrom_dists = vec![None; reader.header().contig_count() as usize];
    let missing_gt: Vec<Allele> = vec![None; PLOIDY];
    for (rn, rec_result) in reader.records().enumerate() {
        let rec = match rec_result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping record {} due to problem: {}", rn, e.to_string());
                continue
            }
        };
        let rid = match rec.rid() {
            Some(x) => x,
            None => continue
        };
        let chr_guider = match guiders.get(rid as usize) {
            Some(x) => match x {
                Some(y) => y,
                None => continue,
            },
            None => continue,
        };
        let cd = match chrom_dists.get_mut(rid as usize) {
            Some(x) => match x {
                Some(y) => y,
                None => {
                    let y = Dists {
                        dist: TriangularMatrix::zeros(samples.len() * PLOIDY),
                        cmps: TriangularMatrix::zeros(samples.len() * PLOIDY)
                    };
                    *x = Some(y);
                    x.as_mut().unwrap()
                },
            },
            None => continue,
        };
        let gts_raw = rec.genotypes()?;
        let mut genotypes: Vec<Vec<Allele>> = Vec::with_capacity(samples.len());
        for i in 0..samples.len() {
            let gt = gts_raw.get(i);
            let alleles: Vec<Allele> = gt.iter().map(get_allele).collect();
            assert!(alleles.len() == PLOIDY, "found genotype of unsopported ploidy in input file");
            genotypes.push(alleles);
        }
        
        let mut out = vec![None; samples.len()];

        zip(genotypes.iter(), out.iter_mut()).enumerate().par_bridge().for_each(|(i, (curr_gt, target))| {
            if curr_gt[1..].iter().all(|x| *x == curr_gt[0]) {
                if curr_gt[0] == None {
                    *target = Some(((curr_gt.clone(), false), 0));
                } else {
                    *target = Some(((curr_gt.clone(), true), 100));
                }
                return
            }

            let sample_guider = &chr_guider[i];
            let mut evidence = Array2::zeros(
                [PLOIDY, max(sample_guider[0].len(), sample_guider[1].len())]
            );
            for t in 0..PLOIDY {
                let guider = &sample_guider[t];
                zip(guider.iter(), evidence.columns_mut().into_iter()).for_each(|((cmp_idx, cmp_val), mut target)| {
                    let cmp_gt = &genotypes[*cmp_idx];
                    for u in 0..PLOIDY {
                        if curr_gt[u] == None {
                            continue;
                        }
                        if cmp_gt.contains(&curr_gt[u]) {
                            // exclusive to diploid
                            let evidence_dest = if t == u { 0 } else { 1 };
                            target[[evidence_dest]] += cmp_val;
                        }
                    }
                })
            }
            let evidence: [f64; PLOIDY] = [evidence.row(0).sum(), evidence.row(1).sum()];

            let mut out_gt = curr_gt.clone();
            // exclusive to diploid
            if evidence[0] == evidence[1] {
                *target = Some(((out_gt, false), 0));
            } else if evidence[0] > evidence[1] {
                let score = (-10. * (1. - evidence[0] / (evidence[0] + evidence[1])).log10()).round();
                *target = Some((
                    (out_gt, true),
                    score.min(100.) as i32
                ));
            } else {
                let score = (-10. * (1. - evidence[1] / (evidence[0] + evidence[1])).log10()).round();
                out_gt.reverse();
                *target = Some((
                    (out_gt, true),
                    score.min(100.) as i32
                ));
            }
        });

        let mut new_rec = writer.empty_record();

        copy_record_stuff(&rec, &mut new_rec);

        let (out_genotypes, phasing_scores): (Vec<(Vec<Allele>, bool)>, Vec<i32>) = out.into_iter().map(|x| x.unwrap()).unzip();
        let gts_flat = out_genotypes.iter().map(
            |(a, p)| if *p {(*a).iter()} else {missing_gt.iter()}
        ).flatten();

        for (i, gt1) in gts_flat.clone().enumerate() {
            for (j, gt2) in gts_flat.clone().take(i+1).enumerate() {
                if *gt2 == None {
                    continue;
                }
                if gt1 != gt2 {
                    cd.dist[[i, j]] += 1;
                }
                cd.cmps[[i, j]] += 1;
            }
        }

        new_rec.push_genotypes(&out_genotypes.into_iter().map(|(alleles, phased)| to_genotype(&alleles, phased)).flatten().collect::<Vec<bcf::record::GenotypeAllele>>())?;
        new_rec.push_format_integer(b"PQ", &phasing_scores)?;
        writer.write(&new_rec)?;
    }
    Ok(chrom_dists)
}

#[derive(Serialize, Deserialize, Clone)]
struct StatsFile {
    samples: Vec<String>,
    contigs: Vec<String>,
    sample_distances: Option<Vec<Option<Dists>>>,
    phased_distances: Option<Vec<Option<Dists>>>,
    guiders: Option<Vec<Option<Vec<Guider>>>>
}

fn main() {
    let cli = Cli::parse();

    rayon::ThreadPoolBuilder::new().num_threads(cli.threads as usize - 1).build_global().unwrap();

    match &cli.command {
        Some(Commands::GenerateTest { file }) => {
            match make_test(file) {
                Err(e) => {
                    eprintln!("{}", e.to_string());
                },
                _ => {}
            };
        },
        Some(Commands::Run { input_file, output: output_file, stats_file: stat_file, max_guiders, output_type: output }) => {
            eprintln!("starting quick read");

            let (dists, contig_names, sample_names) = quick_read(input_file).expect("error during first read");
            eprintln!("distances calculated");

            if let Some(stats_dest) = stat_file {
                let stats = StatsFile {
                    samples: sample_names.clone(),
                    contigs: contig_names.clone(),
                    sample_distances: Some(dists.clone()),
                    phased_distances: None,
                    guiders: None,
                };
                let json_out = serde_json::to_string(&stats).expect("could not convert stats to json");
                let mut f = File::create(stats_dest).expect("could not create stats file");
                f.write_all(json_out.as_bytes()).expect("error writing to stats file");
            }

            let guiders = gen_guiders(&dists, match max_guiders {
                Some(_x @ 0) => None,
                Some(x) => Some(*x),
                None => Some(5)
            });
            eprintln!("guiders generated");
            if let Some(stats_dest) = stat_file {
                let stats = StatsFile {
                    samples: sample_names.clone(),
                    contigs: contig_names.clone(),
                    sample_distances: Some(dists.clone()),
                    phased_distances: None,
                    guiders: Some(guiders.clone()),
                };
                let json_out = serde_json::to_string(&stats).expect("could not convert stats to json");
                let mut f = File::create(stats_dest).expect("could not create stats file");
                f.write_all(json_out.as_bytes()).expect("error writing to stats file");
            }

            let out_matrix = write_phased(input_file, output_file,  &output.unwrap_or(VcfOutputType::B), &guiders).expect("error during phasing");
            eprintln!("done phasing");
            if let Some(stats_dest) = stat_file {
                let stats = StatsFile {
                    samples: sample_names,
                    contigs: contig_names,
                    sample_distances: Some(dists),
                    phased_distances: Some(out_matrix),
                    guiders: Some(guiders)
                };
                let json_out = serde_json::to_string(&stats).expect("could not convert stats to json");
                let mut f = File::create(stats_dest).expect("could not create stats file");
                f.write_all(json_out.as_bytes()).expect("error writing to stats file");
            }
            
        }
        None => {}
    }
}
