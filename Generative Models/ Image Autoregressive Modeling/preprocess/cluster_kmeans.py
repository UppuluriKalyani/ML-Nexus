import logging
import joblib
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
import faiss

logger = logging.getLogger(__name__)

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster-num",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--output",
        type=str,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default='faiss'
    )
    parser.add_argument(
        "--shard",
        type=int,
    )
    args = parser.parse_args()
    logger.info(f"Launched with args: {args}")

    return args


if __name__ == '__main__':
    args = get_args()

    features_batch = []
    for i in range(args.shard):
        features_batch.append(np.load(Path(args.output) / f"features_partial_{i}.npy"))
        print(f"features_partial_{i}.npy loaded")
    features_batch = np.concatenate(features_batch)
    print(features_batch.shape)
    
    if args.backend == "sklearn":
        kmeans_model = MiniBatchKMeans(
            n_clusters=args.cluster_num,
            init='k-means++', 
            max_iter=100,      
            batch_size=10000,
            tol=0.0,
            max_no_improvement=100,
            n_init=10,
            reassignment_ratio=0,
            random_state=1,
            verbose=1,
            compute_labels=False,
            init_size=None,
        )
        kmeans_model.fit(features_batch)
    
        # save kmeans model
        joblib.dump(kmeans_model, open(Path(args.output) / f"km_{args.cluster_num // 1000}k.bin", "wb"))

    elif args.backend == "faiss":
        ncentroids = args.cluster_num
        niter = 20
        verbose = True
        d = features_batch.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(features_batch)
        
        np.save(Path(args.output) / f"km_{args.cluster_num // 1000}k.npy", kmeans.centroids)
