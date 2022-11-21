from lib.convert.preprocess.hsa_mirna import use_hsa_mirna_only
from lib.convert.preprocess.invalid_measurement \
    import remove_invalid_measurement_mirna
from lib.convert.preprocess.ngs_collated_mirna import use_ngs_collated_mirna  # NOQA
from lib.convert.preprocess.unused_mirna import remove_unused_mirna  # NOQA
from lib.convert.preprocess.whiten_ import whiten  # NOQA


DEFAULT_FILTERS = (
    remove_invalid_measurement_mirna,
    use_hsa_mirna_only
)
