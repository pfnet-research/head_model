import os

import pandas as pd


_here = os.path.abspath(os.path.dirname(__file__))
_ngs_collated_mirna_file = os.path.join(
    _here, './data/miRList_NGS3Dgene_posicorr.csv')
_mirnas = pd.read_csv(_ngs_collated_mirna_file,
                      delimiter=',')['Name']


def use_ngs_collated_mirna(df):
    return df[df['G_Name'].isin(_mirnas)]
