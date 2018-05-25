from lmds import compute_lmds
import load
import distances as dist


if __name__ == '__main__':
    tracks = load.load()
    lmds_embeddings = compute_lmds(tracks, nl=20, k=4,
                                   distance=dist.original_distance)
