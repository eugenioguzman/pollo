#![feature(isqrt)]
#![feature(iterator_try_collect)]

use std::{borrow::Cow, cmp::max, collections::{BTreeMap, VecDeque}, fs::File, io::Write, iter::zip, path::PathBuf, sync::Mutex, thread::{self, Thread}};
use clap::{Parser, Subcommand, ValueEnum};
use iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelBridge, ParallelIterator};
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
        output_type: Option<VcfOutputType>,
        #[arg(short, long)]
        use_stats: Option<PathBuf>,
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

impl std::ops::Add<Self> for Dists {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Dists {
            dist: self.dist + rhs.dist,
            cmps: self.cmps + rhs.cmps
        }
    }
}


impl std::ops::AddAssign<Self> for Dists {
    fn add_assign(&mut self, rhs: Self) {
        self.dist += rhs.dist;
        self.cmps += rhs.cmps;
    }
}

impl std::ops::AddAssign<&Self> for Dists {
    fn add_assign(&mut self, rhs: &Self) {
        for (a, b) in zip(self.dist.vec.iter_mut(), rhs.dist.vec.iter()) {
            *a += *b;
        }
        for (a, b) in zip(self.cmps.vec.iter_mut(), rhs.cmps.vec.iter()) {
            *a += *b;
        }
    }
}

fn quick_read(file: &PathBuf) -> Result<(Vec<Option<Dists>>, Vec<String>, Vec<String>), QuickReadError> {
    let mut reader = bcf::Reader::from_path(file)?;
    let samples: Vec<String> = bcf::Read::header(&reader).samples().iter().map(|v| String::from_utf8_lossy(v).into_owned()).collect();
    let nsamples = samples.len();
    let cct = CrossCmpTri::new(nsamples);
    let ncontigs = reader.header().contig_count() as usize;
    let chrom_dists = reader.records().enumerate().par_bridge().map(|(rn, rec_result)| {

        let rec = match rec_result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping record {} due to problem: {}", rn, e.to_string());
                return None
            }
        };

        let rid = match rec.rid() {
            Some(x) => x as usize,
            None => return None
        };

        
        let gts_raw = rec.genotypes().expect("unable to read genotypes");
        let mut sample_missing_mask = vec![false; nsamples];
        let ac = rec.allele_count() as usize;
        let mut gts_vec2d = vec![0i8; nsamples * ac];
        for (i, smm) in sample_missing_mask.iter_mut().enumerate().take(nsamples) {
            let gt = gts_raw.get(i);
            let rowidx = i * ac;
            for a in gt.iter() {
                match get_allele(a) {
                    Some(k) => { gts_vec2d[rowidx + (k as usize)] += 1; },
                    None => { *smm = true; },
                }
            };
        }
        

        let (gt_i, gt_j) = cct.get_ij_arrs(&gts_vec2d, ac);
        let (ms_i, ms_j) = cct.get_ij_arrs(&sample_missing_mask, 1);
        let missing_mask = Vec::from_iter(zip(ms_i.iter(), ms_j.iter()).map(|(a,b)| (*a) | (*b)));
        let distances = Vec::from_iter(
            zip(zip(gt_i.chunks_exact(ac), gt_j.chunks_exact(ac)), missing_mask.iter())
            .map(|((a, b), c)| if *c {0} else {zip(a, b).fold(0, |acc, (x, y)| acc + max(*x - *y, 0) as u128)})
        );

        Some((rid, Dists {
            dist: TriangularMatrix{vec: Vec::from_iter(distances), size: nsamples},
            cmps: TriangularMatrix{vec: Vec::from_iter(missing_mask.into_iter().map(|x| if x {0} else {1})), size: nsamples}
        }))
    }).fold(|| vec![None; ncontigs], |mut acc: Vec<Option<Dists>>, incr| {
        if let Some((rid, inc_dists)) = incr {
            match acc.get_mut(rid).unwrap() {
                Some(v) => *v += inc_dists,
                None => *acc.get_mut(rid).unwrap() = Some(inc_dists),
            };
        }
        acc
    }).reduce(|| vec![None; ncontigs], |mut acc, incr| {
        for (target_dists, incr_dists) in zip(acc.iter_mut(), incr.iter()) {
            if let Some(targ_d) = target_dists {
                if let Some(incr_d) = incr_dists {
                    *targ_d += incr_d
                }
            } else {
                *target_dists = incr_dists.clone();
            }
        }
        acc
    });

    let mut chrom_names = Vec::with_capacity(chrom_dists.len());
    for i in 0..chrom_dists.len() {
        chrom_names.push(String::from_utf8_lossy(reader.header().rid2name(i.try_into()?).unwrap()).into_owned());
    }
    Ok((chrom_dists, chrom_names, samples))
}


struct CrossCmpTri {
    i_viewer: Vec<usize>,
    j_viewer: Vec<usize>
}

impl CrossCmpTri {
    fn new(nsamples: usize) -> CrossCmpTri {
        let (iv, jv) = (0..triangular_matrix_len(nsamples)).map(triangular_matrix_ij).unzip();
        CrossCmpTri {
            i_viewer: iv,
            j_viewer: jv
        }
    }

    #[inline]
    fn get_ij_arrs<T>(&self, row: &[T], stride: usize) -> (Vec<T>, Vec<T>)
    where
        T: Clone,
    {
        let i = Vec::from_iter(self.i_viewer.iter().flat_map(|x| (0..stride).map(|z| row[*x * stride + z].clone())));
        let j = Vec::from_iter(self.j_viewer.iter().flat_map(|x| (0..stride).map(|z| row[*x * stride + z].clone())));
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
        let cct = CrossCmpTri::new(corr.size);
        let mut chr_guiders: Vec<Option<[Vec<(usize, f64)>; 2]>> = vec![None; corr.size];


        chr_guiders.par_iter_mut().enumerate().for_each(|(x, target)| {
            let arr_x = corr.row(x);
            let (arr_xi, arr_xj) = cct.get_ij_arrs(&arr_x, 1);
            let arr_ij = &corr.vec;
            let result = TriangularMatrix {
                vec: zip(zip(arr_xi.iter(), arr_xj.iter()), arr_ij.iter()).map(|((a,b),c)| (**a)*(**b)/(*c)).collect(),
                //arr_ij - arr_xi,
                size: corr.size,
            };
            let ((guider_idx_i, guider_idx_j), _) = result.vec.iter().enumerate()
                .map(|(i, v)| (triangular_matrix_ij(i), *v))
                .filter(|((pi, pj), _)| (*pi != x) && (*pj != x) && (*pi != *pj))
                .fold(
                    ((usize::MAX, usize::MAX), f64::NEG_INFINITY),
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
            let score = Vec::from_iter(zip(zip(arr_i.iter(), arr_j.iter()), arr_x.iter()).map(|((a,b),c)|((*a) - (*b)) * (*c)));
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
    let filt: Vec<bcf::header::Id> = rec.filters().collect();
    let filt: Vec<&bcf::header::Id> = filt.iter().collect();
    new_rec.set_filters(&filt).unwrap();
}

struct MyRecordWriter {
    current_rn: usize,
    queue: VecDeque<Option<bcf::Record>>,
    writer: bcf::Writer,
    parking_lot: BTreeMap<usize, Thread>
}

const WRITER_BUFFER_LIMIT: usize = 1000;
impl MyRecordWriter {
    fn new(writer: bcf::Writer) -> Self {
        MyRecordWriter {
            current_rn: 0,
            queue: VecDeque::new(),
            writer,
            parking_lot: BTreeMap::new()
        }
    }
    fn write(&mut self, rn: usize, record: &bcf::Record) -> Result<(), ()> {
        assert!(rn >= self.current_rn);
        if rn - self.current_rn > WRITER_BUFFER_LIMIT {
            return Err(())
        }
        while self.queue.len() <= rn - self.current_rn {
            self.queue.push_back(None);
        }
        self.queue[rn - self.current_rn] = Some(record.clone());
        while self.queue.front().is_some() && self.queue.front().unwrap().is_some() {
            self.current_rn += 1;
            if let Some(rec) = self.queue.pop_front().unwrap() {
                self.writer.write(&rec).expect("error writing record to output bcf/vcf file");
            }
        }
        for (pos, thread) in self.parking_lot.iter() {
            if pos - self.current_rn < WRITER_BUFFER_LIMIT {
                eprintln!("unparking thread {:?}", thread.name());
                thread.unpark();
            }
        }
        Ok(())
    }

    fn alert_parked(&mut self, rn: usize, thread: Thread) {
        self.parking_lot.insert(rn, thread);
    }

    fn empty_record(&self) -> bcf::Record {
        self.writer.empty_record()
    }
}

fn write_phased(input_file: &PathBuf, output_file: &Option<PathBuf>, output_type: &VcfOutputType, guiders: &[Option<Vec<Guider>>]) -> Result<Vec<Option<Dists>>, rust_htslib::tpool::Error> {
    let mut reader = bcf::Reader::from_path(input_file)?;
    let samples: Vec<String> = reader.header().samples().iter().map(|v| String::from_utf8_lossy(v).into_owned()).collect();
    let nsamples = samples.len();
    let ncontigs = guiders.len();
    let mut header = bcf::Header::from_template(reader.header());
    header.push_record(format!(
        r#"##FORMAT=<ID={},Number={},Type={},Description="{}">"#,
        "PQ", 1, "Integer", "Phasing quality").as_bytes());

    let uncompressed = matches!(output_type, VcfOutputType::V | VcfOutputType::U);
    let htslib_format = match output_type {
        VcfOutputType::V => bcf::Format::Vcf,
        VcfOutputType::Z => bcf::Format::Vcf,
        VcfOutputType::U => bcf::Format::Bcf,
        VcfOutputType::B => bcf::Format::Bcf
    };
    let writer = match output_file {
        Some(f) => bcf::Writer::from_path(f, &header, uncompressed, htslib_format),
        None => bcf::Writer::from_stdout(&header, uncompressed, htslib_format)
    }?;

    let mutex_record_writer = Mutex::new(MyRecordWriter::new(writer));

    let cct = CrossCmpTri::new(nsamples * PLOIDY);

    let chrom_dists = reader.records().enumerate().par_bridge().map(|(rn, rec_result)| {
        let rec = match rec_result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping record {} due to problem: {}", rn, e.to_string());
                return None
            }
        };
        let rid = match rec.rid() {
            Some(x) => x as usize,
            None => return None
        };
        let chr_guider = match guiders.get(rid as usize) {
            Some(Some(x)) => x,
            _ => return None,
        };
        let gts_raw = rec.genotypes().expect("failed to get raw genotypes");
        let mut genotypes: Vec<Vec<Allele>> = Vec::with_capacity(nsamples);
        for i in 0..nsamples {
            let gt = gts_raw.get(i);
            let alleles: Vec<Allele> = gt.iter().map(get_allele).collect();
            assert!(alleles.len() == PLOIDY, "found genotype of unsopported ploidy in input file");
            genotypes.push(alleles);
        }
        
        let mut out = vec![None; nsamples];
        let mut hts_flat = vec![0; nsamples * PLOIDY];
        let mut mis_flat = vec![false; nsamples * PLOIDY];


        for (i, ((curr_gt, target), (hts, mis)))
        in zip(zip(genotypes.iter(), out.iter_mut()), zip(hts_flat.chunks_exact_mut(PLOIDY), mis_flat.chunks_exact_mut(PLOIDY))).enumerate() {
            if curr_gt[1..].iter().all(|x| *x == curr_gt[0]) {
                if curr_gt[0].is_none() {
                    *target = Some(((curr_gt.clone(), false), 0));
                    for x in mis.iter_mut() {
                        *x = true;
                    }
                } else {
                    *target = Some(((curr_gt.clone(), true), 100));
                    for ((ht, mis), oht) in zip(zip(hts.iter_mut(), mis.iter_mut()), curr_gt.iter()) {
                        match oht {
                            Some(x) => *ht = *x,
                            None => *mis = true
                        }
                    }
                }
                continue;
            }

            let sample_guider = &chr_guider[i];
            let mut evidence = vec![0.; PLOIDY * max(sample_guider[0].len(), sample_guider[1].len())];
            for (t, sg) in sample_guider.iter().enumerate() {
                let guider = sg;
                zip(guider.iter(), evidence.chunks_exact_mut(PLOIDY)).for_each(|((cmp_idx, cmp_val), target)| {
                    let cmp_gt = &genotypes[*cmp_idx];
                    for (u, cgt) in curr_gt.iter().enumerate() {
                        if curr_gt[u].is_none() {
                            continue;
                        }
                        if cmp_gt.contains(cgt) {
                            // exclusive to diploid
                            let evidence_dest = if t == u { 0 } else { 1 };
                            target[evidence_dest] += cmp_val;
                        }
                    }
                })
            }
            let evidence: [f64; PLOIDY] = evidence.chunks_exact(PLOIDY).fold([0.; PLOIDY], |acc, x| [acc[0] + x[0], acc[1] + x[1]]);
            // [evidence.row(0).sum(), evidence.row(1).sum()];

            let mut out_gt = curr_gt.clone();
            // exclusive to diploid
            if evidence[0] == evidence[1] {
                *target = Some(((out_gt, false), 0));
                for x in mis.iter_mut() {
                    *x = true;
                }
            } else if evidence[0] > evidence[1] {
                let score = (-10. * (1. - evidence[0] / (evidence[0] + evidence[1])).log10()).round();
                for ((ht, mis), oht) in zip(zip(hts.iter_mut(), mis.iter_mut()), out_gt.iter()) {
                    match oht {
                        Some(x) => *ht = *x,
                        None => *mis = true
                    }
                }
                *target = Some((
                    (out_gt, true),
                    score.min(100.) as i32
                ));
            } else {
                let score = (-10. * (1. - evidence[1] / (evidence[0] + evidence[1])).log10()).round();
                out_gt.reverse();
                for ((ht, mis), oht) in zip(zip(hts.iter_mut(), mis.iter_mut()), out_gt.iter()) {
                    match oht {
                        Some(x) => *ht = *x,
                        None => *mis = true
                    }
                }
                *target = Some((
                    (out_gt, true),
                    score.min(100.) as i32
                ));
            }
        }

        let mut new_rec = mutex_record_writer.lock().unwrap().empty_record();

        copy_record_stuff(&rec, &mut new_rec);

        let (out_genotypes, phasing_scores): (Vec<(Vec<Allele>, bool)>, Vec<i32>) = out.into_iter().map(|x| x.unwrap()).unzip();

        let (ht_i, ht_j) = cct.get_ij_arrs(&hts_flat, 1);
        let (ms_i, ms_j) = cct.get_ij_arrs(&mis_flat, 1);

        let (distances, comparisons): (Vec<u128>, Vec<u128>) = 
            zip(zip(ht_i, ht_j), zip(ms_i, ms_j))
            .map(|((hta, htb), (msa, msb))| 
                (if !msa && !msb && hta != htb {1} else {0}, if !msa && !msb {1} else {0})
            ).unzip();

        new_rec.push_genotypes(
            &out_genotypes.into_iter()
                .flat_map(|(alleles, phased)| to_genotype(&alleles, phased))
                .collect::<Vec<bcf::record::GenotypeAllele>>()
        ).expect("failed to push genotypes to new vcf/bcf record");
        
        new_rec.push_format_integer(b"PQ", &phasing_scores).expect("failed to set PQ format tag for vcf/bcf record");
        while mutex_record_writer.lock().unwrap().write(rn, &new_rec).is_err() {
            let ct = thread::current();
            eprintln!("parking thread {:?}", ct.name());
            mutex_record_writer.lock().unwrap().alert_parked(rn, ct);
            thread::park();
        }
        Some((rid, Dists {
            dist: TriangularMatrix{vec: Vec::from_iter(distances), size: nsamples * 2},
            cmps: TriangularMatrix{vec: Vec::from_iter(comparisons), size: nsamples * 2}
        }))
    }).fold(|| vec![None; ncontigs], |mut acc: Vec<Option<Dists>>, incr| {
        if let Some((rid, inc_dists)) = incr {
            match acc.get_mut(rid).unwrap() {
                Some(v) => *v += inc_dists,
                None => *acc.get_mut(rid).unwrap() = Some(inc_dists),
            };
        }
        acc
    }).reduce(|| vec![None; ncontigs], |mut acc, incr| {
        for (target_dists, incr_dists) in zip(acc.iter_mut(), incr.iter()) {
            if let Some(targ_d) = target_dists {
                if let Some(incr_d) = incr_dists {
                    *targ_d += incr_d
                }
            } else {
                *target_dists = incr_dists.clone();
            }
        }
        acc
    });
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
            if let Err(e) = make_test(file) {
                eprintln!("{}", e.to_string());
            }
        },
        Some(Commands::Run { input_file, output: output_file, stats_file, max_guiders, output_type: output, use_stats }) => {
            let (stats_read_qr, stats_read_g) = if let Some(stats_from) = use_stats {
                let use_stats_file: StatsFile = 
                    serde_json::from_reader(
                        File::open(stats_from).expect("error reading input stats file")
                    ).expect("error parsing input stats file");
                eprintln!("(read stats file)");
                (
                    if let Some(read_sd) = use_stats_file.sample_distances {
                        Some((read_sd, use_stats_file.contigs, use_stats_file.samples))
                    } else {
                        None
                    },
                    use_stats_file.guiders
                )
            } else {
                (None, None)
            };
            
            eprintln!("starting quick read");
            let (dists, contig_names, sample_names) = if let Some(qr_read) = stats_read_qr {
                eprintln!("(using imported distances)");
                qr_read
            } else {
                quick_read(input_file).expect("error during first read")
            };

            eprintln!("distances calculated");

            if let Some(stats_dest) = stats_file {
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


            let guiders = if let Some(g_read) = stats_read_g {
                eprintln!("(using imported guiders)");
                g_read
            } else {
                gen_guiders(&dists, match max_guiders {
                    Some(_x @ 0) => None,
                    Some(x) => Some(*x),
                    None => Some(5)
                })
            };

            eprintln!("guiders generated");
            if let Some(stats_dest) = stats_file {
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
            if let Some(stats_dest) = stats_file {
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
