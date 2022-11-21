UNUSED_FEATURE_NAMES = ('hsa-miR-2467-3p',
                        'hsa-miR-4448',
                        'hsa-miR-4516',
                        'hsa-miR-4525',
                        'hsa-miR-4710',
                        'hsa-miR-4718',
                        'hsa-miR-614',
                        'hsa-miR-8059')


def remove_unused_mirna(df):
    return df[~df['G_Name'].isin(UNUSED_FEATURE_NAMES)]
