
import pstats
#from pstats import SortKey
p = pstats.Stats('qs_profiling')
p.strip_dirs().sort_stats("cumulative").print_stats(100)