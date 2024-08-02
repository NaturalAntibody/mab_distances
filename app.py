import Levenshtein
import pandas as pd
import streamlit as st
from functools import cmp_to_key

from riot_na import create_riot_aa, Scheme, RiotNumberingAA, AirrRearrangementEntryAA
from riot_na.alignment.gene_aligner import AA_ALIGNER_PARAMS
from riot_na.alignment.skbio_alignment import align_aa
from skbio.alignment import StripedSmithWaterman
from functools import partial


@st.cache_resource
def get_riot():
    riot_aa: RiotNumberingAA = create_riot_aa()
    return riot_aa


st.set_page_config(page_title="UAbSim", page_icon=":scales:", layout="wide")


def number_sequence(seq: str, scheme: Scheme) -> AirrRearrangementEntryAA:
    airr_result: AirrRearrangementEntryAA = get_riot().run_on_sequence(
        header="foo",
        query_sequence=seq,
        scheme=scheme,
        extend_alignment=False
    )
    return airr_result


def position_comparer(pos1, pos2, is_imgt=True):
    pos1 = pos1.Index
    pos2 = pos2.Index
    if pos1 == pos2:
        return 0
    pos1, ins1 = pos1.split('.') if '.' in pos1 else (pos1, 0)
    pos1 = int(pos1)
    ins1 = int(ins1)
    pos2, ins2 = pos2.split('.') if '.' in pos2 else (pos2, 0)
    pos2 = int(pos2)
    ins2 = int(ins2)
    if pos1 != pos2:
        return pos1 - pos2
    if pos1 in [33, 61, 112] and is_imgt:
        return ins2 - ins1
    return ins1 - ins2


def get_table(map1, map2, scheme):
    df1 = pd.DataFrame.from_dict(map1, orient='index', columns=['position seq 1'])
    df2 = pd.DataFrame.from_dict(map2, orient='index', columns=['position seq 2'])
    df = df1.join(df2, how='outer')
    df['the same'] = df['position seq 1'] == df['position seq 2']
    list_of_tuples = list(df.itertuples(index=True))
    list_sorted = sorted(list_of_tuples, key=cmp_to_key(partial(position_comparer, is_imgt=scheme == Scheme.IMGT)))
    return pd.DataFrame(list_sorted, columns=['Index', 'seq 1', 'seq 2', 'the same']).set_index('Index')


st.header('Ultimate Antibody Similarity Calculator')

seq1 = st.text_input('Enter 1st sequence',
                     'EVQLVESGGGLVQPGGSLRLSCAASGRTFSYNPMGWFRQAPGKGRELVAAISRTGGSTYYPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCAAAGVRAEDGRVRTLPSEYTFWGQGTQVTVSS')

seq1_numbered = {
    scheme: number_sequence(seq1, scheme) for scheme in Scheme
}
ps1 = seq1_numbered['imgt'].sequence_alignment_aa

st.markdown(f"Primary sequence: `{ps1}`")

seq2 = st.text_input('Enter 2nd sequence',
                     'EVQLVESGGGLVQPGGSLRLSCAYYASGRTFSYNPMGWFRQAPGKGRELVAAISRTGGSTYYPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCAYYAAGVRAEDGRVRTLPSEYTFWGQGTQVTVSS')

seq2_numbered = {
    scheme: number_sequence(seq2, scheme) for scheme in Scheme
}
ps2 = seq2_numbered['imgt'].sequence_alignment_aa
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

numbered_df = {
    s: get_table(seq1_numbered[s].scheme_residue_mapping, seq2_numbered[s].scheme_residue_mapping, s) for s in
    seq1_numbered.keys()
}

same_aa = {s: numbered_df[s]['the same'].sum() for s in numbered_df.keys()}
nuniqe_pos = {s: numbered_df[s].shape[0] for s in numbered_df.keys()}
diff_aa = {s: nuniqe_pos[s] - same_aa[s] for s in numbered_df.keys()}
lev_dist = Levenshtein.distance(ps1, ps2)

aligner = StripedSmithWaterman(ps1, **AA_ALIGNER_PARAMS)

stats = {
    'shortest primary sequence length': min_len,
    'longest primary sequence length': max_len,
    'average primary sequence length': avg_len,
    'number of unique IMGT positions in both sequences': nuniqe_pos['imgt'],
    'number of unique KABAT positions in both sequences': nuniqe_pos['kabat'],
    'number of unique Chothia positions in both sequences': nuniqe_pos['chothia'],
    'number of unique Martin positions in both sequences': nuniqe_pos['martin'],
}
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader('Aligned numbered sequences')
    col11, col12 = col1.columns(2)
    tabs = col11.tabs(['IMGT', 'Chothia', 'Martin', 'Kabat'])
    tabs[0].dataframe(numbered_df['imgt'], height=700)
    tabs[1].dataframe(numbered_df['chothia'], height=700)
    tabs[2].dataframe(numbered_df['martin'], height=700)
    tabs[3].dataframe(numbered_df['kabat'], height=700)

    distances = {
        'Same IMGT': same_aa['imgt'],
        'Different IMGT ': diff_aa['imgt'],

        'Same Kabat': same_aa['kabat'],
        'Different Kabat ': diff_aa['kabat'],

        'Same Chothia': same_aa['chothia'],
        'Different Chothia ': diff_aa['chothia'],

        'Same Martin': same_aa['martin'],
        'Different Martin ': diff_aa['martin'],
    }

    col12.dataframe(pd.DataFrame.from_dict(distances, orient='index', columns=['Value']))

with col2:
    st.subheader('Lengths')
    st.dataframe(pd.DataFrame.from_dict(stats, orient='index', columns=['Value']))

with col3:
    st.subheader('Scheme based distances')
    distances = {
        'IMGT similarity (norm to shortest)': same_aa['imgt'] / min_len,
        'IMGT similarity (norm to longest)': same_aa['imgt'] / max_len,
        'IMGT similarity (norm to average)': same_aa['imgt'] / avg_len,
        'IMGT similarity (norm to num unique IMGT)': same_aa['imgt'] / nuniqe_pos['imgt'],

        'Kabat similarity (norm to shortest)': same_aa['kabat'] / min_len,
        'Kabat similarity (norm to longest)': same_aa['kabat'] / max_len,
        'Kabat similarity (norm to average)': same_aa['kabat'] / avg_len,
        'Kabat similarity (norm to num unique Kabat)': same_aa['kabat'] / nuniqe_pos['kabat'],

        'Chothia similarity (norm to shortest)': same_aa['chothia'] / min_len,
        'Chothia similarity (norm to longest)': same_aa['chothia'] / max_len,
        'Chothia similarity (norm to average)': same_aa['chothia'] / avg_len,
        'Chothia similarity (norm to num unique Chothia)': same_aa['chothia'] / nuniqe_pos['chothia'],

        'Martin similarity (norm to shortest)': same_aa['martin'] / min_len,
        'Martin similarity (norm to longest)': same_aa['martin'] / max_len,
        'Martin similarity (norm to average)': same_aa['martin'] / avg_len,
        'Martin similarity (norm to num unique Martin)': same_aa['martin'] / nuniqe_pos['martin'],
    }

    st.dataframe(pd.DataFrame.from_dict(distances, orient='index', columns=['Value']))

with col4:
    st.subheader('Sequence based distances')
    distances = {
        'SSW aligner': align_aa(aligner, ps1, 'dupa', ps2, 1).seq_identity,
        'Levenshtein ratio': Levenshtein.ratio(ps1, ps2),
        'Jaro similarity': Levenshtein.jaro(ps1, ps2),
        'Jaro-Winkler similarity': Levenshtein.jaro_winkler(ps1, ps2),
        'Levenshtein distance': lev_dist,
        'Levenshtein similarity (norm to shortest)': (1 - lev_dist / min_len),
        'Levenshtein similarity (norm to longest)': (1 - lev_dist / max_len),
        'Levenshtein similarity (norm to average)': (1 - lev_dist / avg_len),

    }

    st.dataframe(pd.DataFrame.from_dict(distances, orient='index', columns=['Value']))
