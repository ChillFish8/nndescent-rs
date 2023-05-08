use std::path::Path;
use std::time::Instant;
use anyhow::Result;
use hdf5::{Dataset, File};
use nndescent::{Metric, NNDescentBuilder};


fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let data = read_json_dataset("../datasets/fashion-mnist-784-euclidean")?;
    let len = dbg!(data.len());

    let start = Instant::now();
    let descent = NNDescentBuilder::default()
        .metric(Metric::Dot)
        .data(data)
        .try_build()?;

    println!("Done! Took {:?} to build, {:?}/vec", start.elapsed(), len as f32 / start.elapsed().as_secs_f32());

    Ok(())
}


fn read_json_dataset(path: impl AsRef<Path>) -> Result<Vec<Vec<f32>>> {
    let path = path.as_ref().with_extension("hdf5");
    let file = File::open(path)?; // open for reading

    let ds: Dataset = file.dataset("train")?; // open the dataset
    let entries = ds.read_2d::<f32>()?;

    let mut resulting_entries = Vec::with_capacity(entries.len());
    for entry in entries.rows() {
        resulting_entries.push(entry.to_vec());
    }
    Ok(resulting_entries)
}