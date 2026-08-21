
"""
05 - Customer Similarity kNN + K-Means Recommender

This version uses two related ideas:
1. kNN customer similarity: find the nearest/similar customers.
2. K-Means clustering: create customer groups and inspect each group.

Customer similarity is based on more than product holdings:
- product holdings
- numeric customer profile features
- categorical customer profile features
- digital login behavior
- transaction behavior

The recommender is independent from the hybrid model. It reads only raw files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PRODUCTS = ['PL', 'CC', 'HL', 'SA', 'RD', 'MF']
HOLDING_COLS = {'PL':'PL_count','CC':'CC_count','HL':'HL_count','SA':'SA_count','RD':'RD_count','MF':'MF_count'}
PRODUCT_NAME_MAP = {'PL':'Personal Loan','CC':'Credit Card','HL':'Home Loan','SA':'Savings Account','RD':'Recurring Deposit','MF':'Mutual Fund'}

# Controls how much each group affects customer similarity.
# These are similarity weights, not product recommendation weights.
FEATURE_GROUP_WEIGHTS = {
    'product_holdings': 0.35,
    'customer_numeric_profile': 0.25,
    'customer_categorical_profile': 0.20,
    'digital_behavior': 0.10,
    'transaction_behavior': 0.10,
}


def parse_args():
    p = argparse.ArgumentParser(description='Customer similarity kNN + K-Means recommender')
    p.add_argument('--raw-dir', default=None, help='Raw data folder, usually data/01_raw')
    p.add_argument('--conversion-data', default=None, help='Optional July conversion CSV for evaluation')
    p.add_argument('--output-dir', default=None, help='Output folder')
    p.add_argument('--top-n', type=int, default=3)
    p.add_argument('--n-neighbors', type=int, default=50)
    p.add_argument('--n-clusters', type=int, default=6)
    p.add_argument('--knn-weight', type=float, default=0.70)
    p.add_argument('--cluster-weight', type=float, default=0.30)
    p.add_argument('--max-onehot-cardinality', type=int, default=40)
    p.add_argument('--visualization-sample-size', type=int, default=3000)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def resolve_paths(args):
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    raw_candidates = []
    if args.raw_dir:
        raw_candidates.append(Path(args.raw_dir))
    raw_candidates += [cwd/'data'/'01_raw', cwd.parent/'data'/'01_raw', script_dir.parent/'data'/'01_raw']
    raw_dir = next((p.resolve() for p in raw_candidates if p.exists()), None)
    if raw_dir is None:
        raise FileNotFoundError('Could not find data/01_raw. Pass --raw-dir.')

    conv_candidates = []
    if args.conversion_data:
        conv_candidates.append(Path(args.conversion_data))
    conv_candidates += [raw_dir/'conversion_data_july_2026.csv', cwd/'data'/'01_raw'/'conversion_data_july_2026.csv']
    conversion_path = next((p.resolve() for p in conv_candidates if p.exists()), None)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else raw_dir.parent/'05_customer_similarity_cf_outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, conversion_path, output_dir


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def add_month_period(df):
    df = df.copy()
    if 'Month period' in df.columns:
        df['Month period'] = df['Month period'].astype(str)
    elif 'Year' in df.columns and 'Month' in df.columns:
        df['Month period'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
    elif 'Date' in df.columns:
        dt = pd.to_datetime(df['Date'], errors='coerce')
        df['Month period'] = dt.dt.to_period('M').astype(str)
    else:
        df['Month period'] = 'unknown'
    return df


def load_customer_table(raw_dir):
    holdings = read_csv(raw_dir/'product_holdings.csv')
    demo = read_csv(raw_dir/'customer_demographics.csv')
    login = read_csv(raw_dir/'digital_login.csv')
    txn = read_csv(raw_dir/'transaction_aggregates.csv')
    if holdings.empty:
        raise FileNotFoundError('product_holdings.csv is required')

    holdings = add_month_period(holdings)
    latest_month = sorted(holdings['Month period'].dropna().astype(str).unique())[-1]
    holdings = holdings[holdings['Month period'].astype(str) == latest_month].copy()

    for prod, col in HOLDING_COLS.items():
        if col not in holdings.columns:
            holdings[col] = 0
        holdings[col] = pd.to_numeric(holdings[col], errors='coerce').fillna(0)
        holdings[prod] = (holdings[col] > 0).astype(int)

    base = holdings[['Customer ID','Month period'] + PRODUCTS].drop_duplicates('Customer ID').copy()

    if not demo.empty and 'Customer ID' in demo.columns:
        demo = add_month_period(demo)
        if latest_month in set(demo['Month period'].dropna().astype(str)):
            demo = demo[demo['Month period'].astype(str) == latest_month].copy()
        demo = demo.drop_duplicates('Customer ID').drop(columns=['Month period'], errors='ignore')
        base = base.merge(demo, on='Customer ID', how='left')

    if not login.empty:
        login = add_month_period(login)
        if 'CustID' in login.columns:
            login = login.rename(columns={'CustID':'Customer ID'})
        nums = [c for c in login.select_dtypes(include=[np.number]).columns if c not in ['Month','Year']]
        if 'Customer ID' in login.columns and nums:
            g = login.groupby('Customer ID', as_index=False)[nums].mean()
            g = g.rename(columns={c:f'digital_{c}' for c in nums})
            base = base.merge(g, on='Customer ID', how='left')

    if not txn.empty:
        txn = add_month_period(txn)
        if 'CustID' in txn.columns:
            txn = txn.rename(columns={'CustID':'Customer ID'})
        nums = [c for c in txn.select_dtypes(include=[np.number]).columns if c not in ['Month','Year']]
        if 'Customer ID' in txn.columns and nums:
            g = txn.groupby('Customer ID', as_index=False)[nums].mean()
            g = g.rename(columns={c:f'txn_{c}' for c in nums})
            base = base.merge(g, on='Customer ID', how='left')

    return base


def scale_block(df, cols):
    if not cols:
        return pd.DataFrame(index=df.index)
    x = df[cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    x = x.replace([np.inf,-np.inf], np.nan).fillna(x.median(numeric_only=True)).fillna(0)
    return pd.DataFrame(StandardScaler().fit_transform(x), index=df.index, columns=cols)


def build_feature_matrix(customer_df, max_onehot_cardinality):
    id_cols = {'Customer ID','Month period'}
    holdings_cols = [c for c in PRODUCTS if c in customer_df.columns]

    numeric = [c for c in customer_df.select_dtypes(include=[np.number]).columns if c not in holdings_cols and c not in id_cols]
    digital_cols = [c for c in numeric if c.startswith('digital_')]
    txn_cols = [c for c in numeric if c.startswith('txn_')]
    numeric_profile_cols = [c for c in numeric if c not in digital_cols and c not in txn_cols]

    cat_cols = []
    for c in customer_df.select_dtypes(include=['object','category']).columns:
        if c in id_cols:
            continue
        nunique = customer_df[c].nunique(dropna=True)
        if 1 < nunique <= max_onehot_cardinality:
            cat_cols.append(c)

    blocks = {
        'product_holdings': customer_df[holdings_cols].fillna(0).astype(float),
        'customer_numeric_profile': scale_block(customer_df, numeric_profile_cols),
        'digital_behavior': scale_block(customer_df, digital_cols),
        'transaction_behavior': scale_block(customer_df, txn_cols),
    }
    if cat_cols:
        blocks['customer_categorical_profile'] = pd.get_dummies(customer_df[cat_cols].fillna('Missing').astype(str)).astype(float)
    else:
        blocks['customer_categorical_profile'] = pd.DataFrame(index=customer_df.index)

    weighted_blocks = {}
    meta = []
    for group, block in blocks.items():
        weight = FEATURE_GROUP_WEIGHTS[group]
        if block.empty:
            meta.append({'feature_group':group,'feature_count':0,'group_weight':weight,'columns_used':''})
            continue
        wb = block * np.sqrt(weight)
        wb.columns = [f'{group}__{c}' for c in wb.columns]
        weighted_blocks[group] = wb
        meta.append({'feature_group':group,'feature_count':block.shape[1],'group_weight':weight,'columns_used':', '.join(map(str, block.columns[:80]))})

    X = pd.concat(weighted_blocks.values(), axis=1)
    X.insert(0, 'Customer ID', customer_df['Customer ID'].values)
    return X, weighted_blocks, pd.DataFrame(meta)


def row_cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den else 0.0


def run_knn(X_df, weighted_blocks, product_matrix, n_neighbors):
    X = X_df.set_index('Customer ID').loc[product_matrix.index].values.astype(float)
    H = product_matrix.values.astype(float)
    ids = product_matrix.index.to_numpy()
    k = min(n_neighbors + 1, len(ids))
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    group_arrays = {g: b.values.astype(float) for g,b in weighted_blocks.items()}
    scores = np.zeros_like(H, dtype=float)
    neighbor_rows = []
    signal_rows = []

    for i, cid in enumerate(ids):
        neigh = ind[i]
        d = dist[i]
        mask = neigh != i
        neigh = neigh[mask]
        sims = np.clip(1 - d[mask], 0, None)
        if sims.sum() > 0:
            scores[i] = sims.reshape(1,-1).dot(H[neigh]).ravel() / sims.sum()

        avg_sim = float(sims.mean()) if len(sims) else 0.0
        top_sim = float(sims[0]) if len(sims) else 0.0
        for pidx, prod in enumerate(PRODUCTS):
            support = H[neigh, pidx] if len(neigh) else np.array([])
            support_count = int(support.sum()) if len(support) else 0
            avg_sim_support = float(sims[support == 1].mean()) if len(sims) and support.sum() else 0.0
            has_product = int(H[i,pidx])
            signal_rows.append({
                'Customer ID': cid,
                'product_code': prod,
                'Product_Name': PRODUCT_NAME_MAP[prod],
                'has_product': has_product,
                'eligible_for_recommendation': int(has_product == 0),
                'knn_customer_similarity_score': float(scores[i,pidx]),
                'knn_recommendation_score': float(scores[i,pidx]) if has_product == 0 else 0.0,
                'neighbor_support_count': support_count,
                'avg_neighbor_similarity': avg_sim,
                'top_neighbor_similarity': top_sim,
                'avg_similarity_of_neighbors_holding_product': avg_sim_support,
            })

        for rank, (j, sim) in enumerate(zip(neigh[:5], sims[:5]), start=1):
            row = {'Customer ID':cid,'neighbor_rank':rank,'neighbor_customer_id':ids[j],'overall_cosine_similarity':float(sim)}
            for g, arr in group_arrays.items():
                row[f'{g}_cosine_similarity'] = row_cosine(arr[i], arr[j]) if arr.shape[1] else np.nan
            neighbor_rows.append(row)

    excluded = np.where(H == 1, 0.0, scores)
    return (
        pd.DataFrame(excluded, index=product_matrix.index, columns=PRODUCTS).reset_index(),
        pd.DataFrame(neighbor_rows),
        pd.DataFrame(signal_rows),
    )


def run_kmeans(X_df, customer_df, n_clusters, seed):
    X = X_df.set_index('Customer ID').values.astype(float)
    ids = X_df['Customer ID'].values
    k = min(n_clusters, len(ids))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    clusters = pd.DataFrame({'Customer ID':ids, 'cluster_id':labels})
    return customer_df.merge(clusters, on='Customer ID', how='left'), pd.DataFrame(km.cluster_centers_).assign(cluster_id=range(k))


def summarize_clusters(clustered):
    rows = []
    numeric_cols = ['Age','Monthly income proxy (sn)','Risk profile score','Days since last login (sn)','Campaign exposure count 90D (sn)','Products ignored count 90D (sn)']
    cat_cols = ['Persona','Lifestage','Income band (sn)','Tier Map','Occupation','Preferred channel (sn)','Gender']
    for cid, g in clustered.groupby('cluster_id'):
        row = {'cluster_id':cid,'customer_count':len(g),'customer_share':len(g)/len(clustered)}
        for p in PRODUCTS:
            row[f'{p}_holding_rate'] = g[p].mean()
        for c in numeric_cols:
            if c in g.columns:
                row[f'avg_{c}'] = pd.to_numeric(g[c], errors='coerce').mean()
        for c in cat_cols:
            if c in g.columns:
                m = g[c].dropna().astype(str).mode()
                top = m.iloc[0] if not m.empty else ''
                row[f'top_{c}'] = top
                row[f'top_{c}_share'] = g[c].astype(str).eq(top).mean() if top else np.nan
        rows.append(row)
    profiles = pd.DataFrame(rows).sort_values('cluster_id')
    rate_cols = [f'{p}_holding_rate' for p in PRODUCTS]
    wide = profiles[['cluster_id','customer_count'] + rate_cols].copy()
    long = wide.melt(id_vars=['cluster_id','customer_count'], value_vars=rate_cols, var_name='product_code', value_name='cluster_holding_rate')
    long['product_code'] = long['product_code'].str.replace('_holding_rate','',regex=False)
    long['Product_Name'] = long['product_code'].map(PRODUCT_NAME_MAP)
    return profiles, wide, long


def score_with_clusters(knn_signal, clustered, cluster_rates_long, knn_weight, cluster_weight):
    out = knn_signal.merge(clustered[['Customer ID','cluster_id']].drop_duplicates(), on='Customer ID', how='left')
    out = out.merge(cluster_rates_long[['cluster_id','product_code','cluster_holding_rate']], on=['cluster_id','product_code'], how='left')
    out['cluster_product_affinity_score'] = out['cluster_holding_rate'].fillna(0.0)
    out['cluster_recommendation_score'] = np.where(out['eligible_for_recommendation'].eq(1), out['cluster_product_affinity_score'], 0.0)
    total = knn_weight + cluster_weight
    kw = knn_weight / total if total > 0 else 0.70
    cw = cluster_weight / total if total > 0 else 0.30
    out['knn_cluster_blended_score'] = np.where(out['eligible_for_recommendation'].eq(1), kw*out['knn_recommendation_score'] + cw*out['cluster_recommendation_score'], 0.0)
    out['knn_weight_used'] = kw
    out['cluster_weight_used'] = cw
    return out


def make_topn(scored, score_col, model_name, top_n):
    c = scored[scored['eligible_for_recommendation'].eq(1)].copy()
    c = c.sort_values(['Customer ID', score_col], ascending=[True, False])
    c['recommendation_rank'] = c.groupby('Customer ID').cumcount() + 1
    c = c[c['recommendation_rank'] <= top_n].copy()
    c['recommendation_model'] = model_name
    cols = ['Customer ID','Month period','cluster_id','recommendation_model','recommendation_rank','product_code','Product_Name',score_col,'knn_recommendation_score','cluster_recommendation_score','neighbor_support_count','avg_neighbor_similarity','top_neighbor_similarity','cluster_product_affinity_score']
    return c[[x for x in cols if x in c.columns]]


def evaluate(top_recs, conversion_path, top_n, model_name):
    if conversion_path is None:
        return {'model':model_name, 'note':'No conversion data supplied'}
    conv = pd.read_csv(conversion_path)
    if 'Product converted' not in conv.columns:
        return {'model':model_name, 'note':'Conversion data missing Product converted'}
    conv = conv[['Customer ID','Product converted']].drop_duplicates()
    j = conv.merge(top_recs[['Customer ID','product_code','recommendation_rank']], left_on=['Customer ID','Product converted'], right_on=['Customer ID','product_code'], how='left')
    ranks = j['recommendation_rank'].dropna()
    return {
        'model': model_name,
        'converted_customers': conv['Customer ID'].nunique(),
        'evaluated_conversion_rows': len(conv),
        'hit_rate_at_1': (j['recommendation_rank'] == 1).sum()/len(conv) if len(conv) else np.nan,
        f'hit_rate_at_{top_n}': j['recommendation_rank'].le(top_n).sum()/len(conv) if len(conv) else np.nan,
        'mrr': (1/ranks).sum()/len(conv) if len(ranks) else 0.0,
        'avg_rank_if_hit': ranks.mean() if len(ranks) else np.nan,
    }


def create_visuals(outdir, feature_meta, clustered, cluster_rates_long, scored, top_blended, eval_df, X_df, seed, sample_size):
    viz = outdir/'visualizations'
    viz.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8,4)); m = feature_meta.sort_values('group_weight')
    plt.barh(m['feature_group'], m['group_weight']); plt.xlabel('Feature group weight'); plt.title('Customer similarity feature group weights')
    plt.tight_layout(); plt.savefig(viz/'01_feature_group_weights.png', dpi=150); plt.close()

    plt.figure(figsize=(8,4)); counts = clustered['cluster_id'].value_counts().sort_index()
    plt.bar(counts.index.astype(str), counts.values); plt.xlabel('Cluster ID'); plt.ylabel('Customers'); plt.title('K-Means customer cluster sizes')
    plt.tight_layout(); plt.savefig(viz/'02_cluster_sizes.png', dpi=150); plt.close()

    pivot = cluster_rates_long.pivot(index='cluster_id', columns='product_code', values='cluster_holding_rate')[PRODUCTS]
    plt.figure(figsize=(8,4)); plt.imshow(pivot.values, aspect='auto'); plt.xticks(range(len(PRODUCTS)), PRODUCTS); plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
    plt.xlabel('Product'); plt.ylabel('Cluster ID'); plt.title('Cluster product holding rates'); plt.colorbar(label='Holding rate')
    plt.tight_layout(); plt.savefig(viz/'03_cluster_product_holding_rates.png', dpi=150); plt.close()

    plt.figure(figsize=(8,4)); vals = scored.loc[scored['eligible_for_recommendation'].eq(1), 'knn_recommendation_score']
    plt.hist(vals, bins=30); plt.xlabel('kNN recommendation score'); plt.ylabel('Customer-product rows'); plt.title('Distribution of kNN signal scores')
    plt.tight_layout(); plt.savefig(viz/'04_knn_signal_score_distribution.png', dpi=150); plt.close()

    plt.figure(figsize=(8,4)); mix = top_blended['product_code'].value_counts().reindex(PRODUCTS).fillna(0)
    plt.bar(mix.index, mix.values); plt.xlabel('Product'); plt.ylabel('Top-3 recommendation count'); plt.title('Recommendation mix: kNN + K-Means blended')
    plt.tight_layout(); plt.savefig(viz/'05_recommendation_mix_blended.png', dpi=150); plt.close()

    metric_col = 'hit_rate_at_3'
    if metric_col in eval_df.columns:
        e = eval_df.dropna(subset=[metric_col]).copy()
        if len(e):
            plt.figure(figsize=(8,4)); plt.barh(e['model'], e[metric_col]); plt.xlabel('Hit Rate@3'); plt.title('Model accuracy against July conversions')
            plt.tight_layout(); plt.savefig(viz/'06_accuracy_comparison.png', dpi=150); plt.close()

    X = X_df.set_index('Customer ID')
    n = min(sample_size, len(X))
    if n >= 2 and X.shape[1] >= 2:
        ids = X.sample(n=n, random_state=seed).index
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            coords = PCA(n_components=2, random_state=seed).fit_transform(X.loc[ids].values)
        labs = clustered.set_index('Customer ID').loc[ids, 'cluster_id']
        plt.figure(figsize=(7,5))
        for cl in sorted(labs.dropna().unique()):
            mask = labs.values == cl
            plt.scatter(coords[mask,0], coords[mask,1], s=10, alpha=0.7, label=f'Cluster {cl}')
        plt.xlabel('PCA component 1'); plt.ylabel('PCA component 2'); plt.title('Customer clusters in 2D PCA view'); plt.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(viz/'07_pca_customer_clusters.png', dpi=150); plt.close()


def run(args):
    raw_dir, conv_path, out = resolve_paths(args)
    print('Reading raw data from:', raw_dir)
    print('Writing outputs to:', out)

    customers = load_customer_table(raw_dir)
    customers.to_csv(out/'customer_similarity_model_input_table.csv', index=False)

    X_df, blocks, meta = build_feature_matrix(customers, args.max_onehot_cardinality)
    X_df.to_csv(out/'customer_similarity_feature_matrix.csv', index=False)
    X_df.head(1000).to_csv(out/'customer_similarity_feature_matrix_sample.csv', index=False)
    meta.to_csv(out/'customer_similarity_feature_metadata.csv', index=False)

    product_matrix = customers.set_index('Customer ID')[PRODUCTS].fillna(0).astype(float)
    product_matrix.reset_index().to_csv(out/'customer_product_holdings_matrix.csv', index=False)

    knn_matrix, neighbors, knn_signal = run_knn(X_df, blocks, product_matrix, args.n_neighbors)
    knn_matrix.to_csv(out/'knn_customer_similarity_score_matrix.csv', index=False)
    neighbors.to_csv(out/'customer_similarity_neighbor_sample.csv', index=False)
    knn_signal.to_csv(out/'knn_signal_by_customer_product.csv', index=False)

    clustered, centers = run_kmeans(X_df, customers, args.n_clusters, args.seed)
    clustered.to_csv(out/'customer_clusters.csv', index=False)
    centers.to_csv(out/'kmeans_cluster_centers.csv', index=False)
    profiles, rates_wide, rates_long = summarize_clusters(clustered)
    profiles.to_csv(out/'cluster_profiles.csv', index=False)
    rates_wide.to_csv(out/'cluster_product_holding_rates_wide.csv', index=False)
    rates_long.to_csv(out/'cluster_product_holding_rates_long.csv', index=False)

    scored = score_with_clusters(knn_signal, clustered, rates_long, args.knn_weight, args.cluster_weight)
    scored.to_csv(out/'customer_similarity_all_scores.csv', index=False)

    top_knn = make_topn(scored, 'knn_recommendation_score', 'kNN Customer Similarity', args.top_n)
    top_cluster = make_topn(scored, 'cluster_recommendation_score', 'K-Means Cluster Product Affinity', args.top_n)
    top_blended = make_topn(scored, 'knn_cluster_blended_score', 'kNN + K-Means Blended', args.top_n)
    top_knn.to_csv(out/'final_top_3_recommendations_knn_customer_similarity.csv', index=False)
    top_cluster.to_csv(out/'final_top_3_recommendations_cluster_based.csv', index=False)
    top_blended.to_csv(out/'final_top_3_recommendations_knn_cluster_blended.csv', index=False)

    eval_df = pd.DataFrame([
        evaluate(top_knn, conv_path, args.top_n, 'kNN Customer Similarity'),
        evaluate(top_cluster, conv_path, args.top_n, 'K-Means Cluster Product Affinity'),
        evaluate(top_blended, conv_path, args.top_n, 'kNN + K-Means Blended'),
    ])
    eval_df.to_csv(out/'customer_similarity_model_accuracy_comparison.csv', index=False)

    sanity = pd.DataFrame([
        {'check':'customers','value':customers['Customer ID'].nunique()},
        {'check':'similarity_feature_columns','value':X_df.shape[1]-1},
        {'check':'n_neighbors','value':args.n_neighbors},
        {'check':'n_clusters','value':args.n_clusters},
        {'check':'knn_top3_rows','value':len(top_knn)},
        {'check':'cluster_top3_rows','value':len(top_cluster)},
        {'check':'blended_top3_rows','value':len(top_blended)},
        {'check':'already_held_positive_blended_score_rows','value':int(((scored['has_product']==1)&(scored['knn_cluster_blended_score']>0)).sum())},
    ])
    sanity.to_csv(out/'customer_similarity_cf_sanity_checks.csv', index=False)

    create_visuals(out, meta, clustered, rates_long, scored, top_blended, eval_df, X_df, args.seed, args.visualization_sample_size)

    print('\nDone. Key outputs:')
    for f in ['final_top_3_recommendations_knn_customer_similarity.csv','final_top_3_recommendations_cluster_based.csv','final_top_3_recommendations_knn_cluster_blended.csv','customer_clusters.csv','cluster_profiles.csv','knn_signal_by_customer_product.csv','customer_similarity_model_accuracy_comparison.csv','visualizations/']:
        print('-', out/f)


if __name__ == '__main__':
    run(parse_args())
