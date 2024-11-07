from pathlib import Path
import shutil

def main(args):
    with open(args.result_path / f"{args.subset}.txt", "w") as fw:
        for i in range(args.num_shards):
            print(f"shard {i}")
            with open(args.result_path / f"{args.subset}_shard_{i}" / f"generate-{args.subset}.txt", "r") as fp:
                for line in fp.readlines():
                    x = line.strip().split("\t")
                    if len(x[0].split("-")) != 2:
                        continue
                    x[0] = x[0].split("-")[0] + "-" + str(int(x[0].split("-")[-1]) * args.num_shards + i)
                    fw.writelines("\t".join(x) + "\n")
    
    for i in range(args.num_shards):
        shutil.rmtree(Path(args.result_path / f"{args.subset}_shard_{i}"))
        
            
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-shards", type=int
    )
    parser.add_argument(
        "--result-path", type=Path
    )
    parser.add_argument(
        "--subset", type=str
    )
    args = parser.parse_args()

    main(args)