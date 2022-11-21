INVALID_FEATURE_NAMES = ('hsa-miR-1290',
                         'hsa-miR-451a',
                         'hsa-miR-5100',
                         'hsa-miR-4448')


def remove_invalid_measurement_mirna(df):
    return df[~df['G_Name'].isin(INVALID_FEATURE_NAMES)]
