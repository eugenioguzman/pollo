#![feature(isqrt)]
#![feature(iterator_try_collect)]

use std::{borrow::Cow, fs::File, io::Write, iter::zip, path::PathBuf};
use clap::{Parser, Subcommand};
use rust_htslib::bcf::{self, Read};
use serde::{Deserialize, Serialize};
use triangular_matrix::TriangularMatrix;
mod triangular_matrix;

const PLOIDY: usize = 2;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    GenerateTest { file: PathBuf },
    Run {
        input_file: PathBuf,
        output_file: PathBuf,
        #[arg(short, long)]
        stats_file: Option<PathBuf>
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


#[derive(PartialEq, Eq, Clone, Debug)]
enum AlleleType {
    Ref,
    Alt(i32),
    Missing
}

impl PartialOrd for AlleleType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(match &self {
            AlleleType::Ref => match &other {
                AlleleType::Ref => std::cmp::Ordering::Equal,
                _ => std::cmp::Ordering::Less,
            },
            AlleleType::Alt(a1) => match &other {
                AlleleType::Ref => std::cmp::Ordering::Greater,
                AlleleType::Alt(a2) => a1.cmp(a2),
                AlleleType::Missing => std::cmp::Ordering::Less,
            },
            AlleleType::Missing => match &other {
                AlleleType::Missing => std::cmp::Ordering::Equal,
                _ => std::cmp::Ordering::Greater,
            }
        })
    }
}

impl Ord for AlleleType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
    
    fn max(self, other: Self) -> Self
    where
        Self: Sized,
    {
        match self.partial_cmp(&other).unwrap() {
            std::cmp::Ordering::Less => other,
            std::cmp::Ordering::Equal => self,
            std::cmp::Ordering::Greater => self,
        }
    }
    
    fn min(self, other: Self) -> Self
    where
        Self: Sized,
    {
        match self.partial_cmp(&other).unwrap() {
            std::cmp::Ordering::Less => self,
            std::cmp::Ordering::Equal => self,
            std::cmp::Ordering::Greater => other,
        }
    }
    
    fn clamp(self, min: Self, max: Self) -> Self
    where
        Self: Sized,
        Self: PartialOrd,
    {
        assert!(match min.partial_cmp(&max).unwrap() {
            std::cmp::Ordering::Greater => false,
            _ => true,
        });

        match self.partial_cmp(&min).unwrap() {
            std::cmp::Ordering::Less => min,
            _ => match self.partial_cmp(&max).unwrap() {
                std::cmp::Ordering::Greater => max,
                _ => self
            }
        }
    }
}

fn get_allele(ga: &bcf::record::GenotypeAllele) -> AlleleType {
    match ga {
        bcf::record::GenotypeAllele::Unphased(x) => {
            if *x == 0 {
                AlleleType::Ref
            } else {
                AlleleType::Alt(*x)
            }
        },
        bcf::record::GenotypeAllele::Phased(x) => {
            if *x == 0 {
                AlleleType::Ref
            } else {
                AlleleType::Alt(*x)
            }
        },
        bcf::record::GenotypeAllele::UnphasedMissing => AlleleType::Missing,
        bcf::record::GenotypeAllele::PhasedMissing => AlleleType::Missing,
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

fn quick_read(file: &PathBuf) -> Result<(Vec<Dists>, Vec<String>, Vec<String>), QuickReadError> {
    let mut reader = bcf::Reader::from_path(file)?;
    let samples: Vec<String> = bcf::Read::header(&reader).samples().iter().map(|v| String::from_utf8_lossy(*v).into_owned()).collect();
    let mut chrom_dists = vec![Dists {
        dist: TriangularMatrix::zeros(samples.len()),
        cmps: TriangularMatrix::zeros(samples.len())
    }; reader.header().contig_count() as usize];
    for (rn, rec_result) in reader.records().enumerate() {
        let rec = match rec_result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping record {} due to problem: {}", rn, e.to_string());
                continue
            }
        };
        let gts_raw = rec.genotypes()?;
        let rid = match rec.rid() {
            Some(x) => x,
            None => continue
        };
        let mut poisoned = vec![false; samples.len()];
        let mut rows = vec![vec![0; rec.allele_count().try_into()?]; samples.len()];
        for i in 0..samples.len() {
            let gt = gts_raw.get(i);
            for a in gt.iter().map(get_allele) {
                match a {
                    AlleleType::Missing => { poisoned[i] = true },
                    AlleleType::Ref => { rows[i][0] += 1 },
                    AlleleType::Alt(k) => { rows[i][usize::try_from(k)?] += 1 }
                }
            }
            let current_row = &rows[i];
            for (j, (p, r)) in zip(poisoned[..=i].iter(), rows[..=i].iter()).enumerate() {
                if !p {
                    let dist = if j != i { zip(current_row.iter(), r.iter()).fold(0i32, |acc, (a, b)| {
                        let diff = *a - *b;
                        return if diff > 0 { diff + acc } else { acc }
                    }) } else { 0 };
                    chrom_dists[rid as usize].dist[[i, j]] += u128::try_from(dist).unwrap();
                    chrom_dists[rid as usize].cmps[[i, j]] += 1;
                }
            }
        }
    }
    let mut chrom_names = Vec::with_capacity(chrom_dists.len());
    for i in 0..chrom_dists.len() {
        chrom_names.push(String::from_utf8_lossy(reader.header().rid2name(i.try_into()?).unwrap()).into_owned());
    }
    Ok((chrom_dists, chrom_names, samples))
}


type Guider = [Vec<(usize, f64)>; PLOIDY];

fn get_guiders(dists: &Vec<Dists>, limit: Option<usize>) -> Vec<Vec<Guider>> {
    let mut guiders = Vec::with_capacity(dists.len());
    for dist in dists {
        let d = dist.clone();
        let mut corr = -d.dist.map(|x| *x as f64) / (d.cmps.map(|x| *x as f64) * (PLOIDY as f64)) + 1f64;
        corr.fillna(0.);
        let corr = corr;
        let mut chr_guiders: Vec<[Vec<(usize, f64)>; 2]> = Vec::with_capacity(corr.size);
        for i in 0..corr.size {
            let row: Vec<f64> = corr.row(i).iter().map(|x| (*x).clone()).collect();
            let mut max_coord = (0usize, 0usize);
            let mut max_value = -f64::INFINITY;
            for j in 0..corr.size {
                if j == i {
                    continue
                }
                for k in 0..j {
                    if k == i {
                        continue
                    }
                    let value = corr[[j, k]] - row[k];
                    if value > max_value {
                        max_value = value;
                        max_coord = (j, k);
                    }
                }
            }
            
            let mut grps: [Vec<(usize, f64)>; PLOIDY] = [Vec::with_capacity(corr.size-1), Vec::with_capacity(corr.size-1)];
            for j in 0..corr.size {
                if j == i {
                    continue
                }
                let value = (corr[[max_coord.0, j]] - corr[[max_coord.1, j]]) * corr[[i, j]];
                // Group into "teams" depending on which is closer
                if value > 0.0 {
                    grps[0].push((j, value));
                } else if value < 0.0 {
                    grps[1].push((j, value));
                }
            }
            
            for p in 0..PLOIDY {
                // Should we truncate the guiders before normalization?
                grps[p].sort_by(|(_, a), (_, b)| (-a).partial_cmp(&(-b)).unwrap_or(std::cmp::Ordering::Equal));
                match limit {
                    Some(x) => grps[p].truncate(x),
                    _ => ()
                };

                // SOFTMAX normalization
                let norm_const = grps[p].iter().fold(0f64, |acc, (_, y)| y.exp() + acc);
                for (_idx, val) in grps[p].iter_mut() {
                    let exp_val = val.exp();
                    *val = exp_val / norm_const;
                }
            }
            let grps = grps;
            chr_guiders.push(grps);
        }
        guiders.push(chr_guiders);
    }
    guiders
}

fn to_genotype(alleles: &Vec<AlleleType>, phased: bool) -> Vec<bcf::record::GenotypeAllele> {
    let mut ordered_alleles = Cow::from(alleles);
    if !phased {
        ordered_alleles.to_mut().sort();
    }
    assert!(ordered_alleles.len() > 0);

    let mut genotype = Vec::with_capacity(ordered_alleles.len());
    
    genotype.push(match ordered_alleles[0] {
        AlleleType::Ref => bcf::record::GenotypeAllele::Unphased(0),
        AlleleType::Alt(i) => bcf::record::GenotypeAllele::Unphased(i),
        AlleleType::Missing => bcf::record::GenotypeAllele::UnphasedMissing,
    });

    for al in ordered_alleles[1..].iter() {
        genotype.push(match al {
            AlleleType::Ref => if phased { bcf::record::GenotypeAllele::Phased(0) } else { bcf::record::GenotypeAllele::Unphased(0) },
            AlleleType::Alt(i) => if phased { bcf::record::GenotypeAllele::Phased(*i) } else { bcf::record::GenotypeAllele::Unphased(*i) },
            AlleleType::Missing => if phased { bcf::record::GenotypeAllele::PhasedMissing } else { bcf::record::GenotypeAllele::UnphasedMissing },
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

fn write_phased(input_file: &PathBuf, output_file: &PathBuf, guiders: Vec<Vec<Guider>>) -> Result<Vec<Dists>, rust_htslib::tpool::Error> {
    let mut reader = bcf::Reader::from_path(input_file)?;
    let samples: Vec<String> = reader.header().samples().iter().map(|v| String::from_utf8_lossy(*v).into_owned()).collect();
    let mut header = bcf::Header::from_template(reader.header());
    header.push_record(format!(
        r#"##FORMAT=<ID={},Number={},Type={},Description="{}">"#,
        "PQ", 1, "Integer", "Phasing quality").as_bytes());
    let mut writer = bcf::Writer::from_path(output_file, &header, true, bcf::Format::Vcf)?;
    let mut chrom_dists = vec![Dists {
        dist: TriangularMatrix::zeros(samples.len() * PLOIDY),
        cmps: TriangularMatrix::zeros(samples.len() * PLOIDY)
    }; reader.header().contig_count() as usize];
    let missing_gt = vec![AlleleType::Missing; PLOIDY];
    for (rn, rec_result) in reader.records().enumerate() {
        let rec = match rec_result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping record {} due to problem: {}", rn, e.to_string());
                continue
            }
        };
        let gts_raw = rec.genotypes()?;
        let rid = match rec.rid() {
            Some(x) => x,
            None => continue
        };
        let guider = &guiders[rid as usize];
        let mut genotypes: Vec<Vec<AlleleType>> = Vec::with_capacity(samples.len());
        for i in 0..samples.len() {
            let gt = gts_raw.get(i);
            let alleles: Vec<AlleleType> = gt.iter().map(get_allele).collect();
            assert!(alleles.len() == PLOIDY, "found genotype of unsopported ploidy in input file");
            genotypes.push(alleles);
        }
        
        let mut out_genotypes = Vec::with_capacity(samples.len());
        let mut phasing_scores = Vec::with_capacity(samples.len());
        for (i, curr_gt) in genotypes.iter().enumerate() {
            if curr_gt[1..].iter().all(|x| *x == curr_gt[0]) {
                if curr_gt[0] == AlleleType::Missing {
                    out_genotypes.push((curr_gt.clone(), false));
                    phasing_scores.push(0);
                } else {
                    out_genotypes.push((curr_gt.clone(), true));
                    phasing_scores.push(100);
                }
                continue
            }
            // should have factorial(PLOIDY) items
            // in this case PLOIDY = 2, so
            // factorial(PLOIDY) = 2 = PLOIDY
            let mut evidence = [0f64; PLOIDY];
            for t in 0..PLOIDY {
                for (cmp_idx, cmp_val) in guider[i][t].iter() {
                    let cmp_gt = &genotypes[*cmp_idx];
                    for u in 0..PLOIDY {
                        if curr_gt[u] == AlleleType::Missing {
                            continue;
                        }
                        if cmp_gt.contains(&curr_gt[u]) {
                            // exclusive to diploid
                            let evidence_dest = if t == u { 0 } else { 1 };
                            evidence[evidence_dest] += cmp_val;
                        }
                    }
                }
            }

            let mut out_gt = curr_gt.clone();
            // exclusive to diploid
            if evidence[0] == evidence[1] {
                out_genotypes.push((out_gt, false));
                phasing_scores.push(0);
            } else if evidence[0] > evidence[1] {
                out_genotypes.push((out_gt, true));
                phasing_scores.push((-10. * (1. - evidence[0] / (evidence[0] + evidence[1])).log10()).round() as i32);
            } else {
                out_gt.reverse();
                out_genotypes.push((out_gt, true));
                phasing_scores.push((-10. * (1. - evidence[1] / (evidence[0] + evidence[1])).log10()).round() as i32);
            }
        }
        let mut new_rec = writer.empty_record();

        copy_record_stuff(&rec, &mut new_rec);

        let gts_flat = out_genotypes.iter().map(
            |(a, p)| if *p {(*a).iter()} else {(missing_gt).iter()}
        ).flatten();

        for (i, gt1) in gts_flat.clone().enumerate() {
            if *gt1 == AlleleType::Missing {
                continue;
            }
            for (j, gt2) in gts_flat.clone().take(i+1).enumerate() {
                if *gt2 == AlleleType::Missing {
                    continue;
                }
                if gt1 != gt2 {
                    chrom_dists[rid as usize].dist[[i, j]] += 1;
                }
                chrom_dists[rid as usize].cmps[[i, j]] += 1;
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
    sample_distances: Option<Vec<Dists>>,
    phased_distances: Option<Vec<Dists>>
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::GenerateTest { file }) => {
            match make_test(file) {
                Err(e) => {
                    eprintln!("{}", e.to_string());
                },
                _ => {}
            };
        },
        Some(Commands::Run { input_file, output_file, stats_file: stat_file }) => {
            let (dists, contig_names, sample_names) = quick_read(input_file).expect("error during first read");
            eprintln!("distances calculated");
            let guiders = get_guiders(&dists, Some(5));
            eprintln!("guiders generated");
            let out_matrix = write_phased(input_file, output_file, guiders).expect("error during phasing");
            eprintln!("done phasing");
            //for (d, name) in zip(out_matrix, contig_names) {
            //    let mut corr = -d.dist.map(|x| *x as f64) / (d.cmps.map(|x| *x as f64)) + 1f64;
            //    corr.fillna(0.);
            //    let corr = corr;
            //    eprintln!("====== {}", name);
            //    eprintln!("{:?}", corr);
            //}
            if let Some(stats_dest) = stat_file {
                let stats = StatsFile {
                    samples: sample_names,
                    contigs: contig_names,
                    sample_distances: Some(dists),
                    phased_distances: Some(out_matrix),
                };
                let json_out = serde_json::to_string(&stats).expect("could not convert stats to json");
                let mut f = File::create(stats_dest).expect("could not create stats file");
                f.write_all(json_out.as_bytes()).expect("error writing to stats file");
            }
            
        }
        None => {}
    }
}
