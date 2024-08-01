import Levenshtein
import pandas as pd
import streamlit as st
from functools import cmp_to_key
from riot_na import create_riot_aa, Scheme, RiotNumberingAA, AirrRearrangementEntryAA


@st.cache_resource
def get_riot():
    riot_aa: RiotNumberingAA = create_riot_aa()
    return riot_aa


st.set_page_config(page_title="UAbDist", page_icon=":scales:", layout="wide")

st.header('Ultimate Antibody Distances')

seq1 = st.text_input('Enter 1st sequence')

airr_result1: AirrRearrangementEntryAA = get_riot().run_on_sequence(
    header="foo",
    query_sequence=seq1,
    scheme=Scheme.IMGT,
    extend_alignment=False
)
ps1 = airr_result1.sequence_alignment_aa

st.markdown(f"Primary sequence: `{ps1}`")

seq2 = st.text_input('Enter 2nd sequence')

airr_result2: AirrRearrangementEntryAA = get_riot().run_on_sequence(
    header="foo",
    query_sequence=seq2,
    scheme=Scheme.IMGT,
    extend_alignment=False
)
ps2 = airr_result2.sequence_alignment_aa
st.markdown(f"Primary sequence: `{ps2}`")

if not ps1 or not ps2:
    st.error('Please enter both sequences')
    st.stop()

len_ps1 = len(ps1)
len_ps2 = len(ps2)
max_len = max(len_ps1, len_ps2)
min_len = min(len_ps1, len_ps2)
avg_len = (len_ps1 + len_ps2) / 2

lev_distance = Levenshtein.distance(ps1, ps2)


def imgt_comparer(pos1, pos2):
    pos1 = pos1.Index
    pos2 = pos2.Index
    if pos1 == pos2:
        return 0
    imgt1, ins1 = pos1.split('.') if '.' in pos1 else (pos1, None)
    imgt1 = int(imgt1)
    ins1 = int(ins1) if ins1 else 0
    imgt2, ins2 = pos2.split('.') if '.' in pos2 else (pos2, None)
    imgt2 = int(imgt2)
    ins2 = int(ins2) if ins2 else 0
    if imgt1 != imgt2:
        return imgt1 - imgt2
    if imgt1 not in [33, 61, 112]:
        return ins1 - ins2
    return ins2 - ins1


def get_imgt_table(map1, map2):
    df1 = pd.DataFrame.from_dict(map1, orient='index', columns=['IMGT seq 1'])
    df2 = pd.DataFrame.from_dict(map2, orient='index', columns=['IMGT seq 2'])
    df = df1.join(df2, how='outer')
    df['the same'] = df['IMGT seq 1'] == df['IMGT seq 2']
    list_of_tuples = list(df.itertuples(index=True))
    list_sorted = sorted(list_of_tuples, key=cmp_to_key(imgt_comparer))
    return pd.DataFrame(list_sorted, columns=['Index', 'IMGT seq 1', 'IMGT seq 2', 'the same']).set_index('Index')


imgt_df = get_imgt_table(airr_result1.scheme_residue_mapping, airr_result2.scheme_residue_mapping)

imgt_same = imgt_df['the same'].sum()
imgt_nunique = imgt_df.shape[0]
imgt_diff = imgt_nunique - imgt_same
lev_dist = Levenshtein.distance(ps1, ps2)

stats = {
    'shortest primary sequence length': min_len,
    'longest primary sequence length': max_len,
    'average primary sequence length': avg_len,
    'number of unique IMGT positions in both sequences': imgt_nunique,
}
col1, col2, col3 = st.columns(3)

with col1:
    st.dataframe(imgt_df, height=700)

with col2:
    st.subheader('Lengths')
    st.dataframe(pd.DataFrame.from_dict(stats, orient='index', columns=['Value']))

with col3:
    st.subheader('Distances')
    distances = {
        'Levenshtein ratio': Levenshtein.ratio(ps1, ps2),
        'Levenshtein distance': lev_dist,
        'Levenshtein similarity (norm to shortest)': (1 - lev_dist / min_len),
        'Levenshtein similarity (norm to longest)': (1 - lev_dist / max_len),
        'Levenshtein similarity (norm to average)': (1 - lev_dist / avg_len),
        'Same IMGT': imgt_same,
        'Different IMGT ': imgt_diff,
        'IMGT similarity (norm to shortest)': imgt_same / min_len,
        'IMGT similarity (norm to longest)': imgt_same / max_len,
        'IMGT similarity (norm to average)': imgt_same / avg_len,
        'IMGT similarity (norm to num unique IMGT)': imgt_same / imgt_nunique,
    }

    st.dataframe(pd.DataFrame.from_dict(distances, orient='index', columns=['Value']), height=700)

# st.write(airr_result)
