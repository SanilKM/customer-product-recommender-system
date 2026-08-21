"""
Postprocess one cluster-first kNN run.

Adds the manager-requested summary outputs:
- exact feature values per cluster
- exact driving features per cluster
- outlier treatment diagnostics
- cumulative Hit@1/2/3
- confusion matrix
- report-ready plots
"""
from __future__ import annotations
import argparse, json, time, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

PRODUCTS = ["PL","CC","HL","SA","RD","MF"]


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--raw-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--dataset-name', default=None)
    p.add_argument('--top-n', type=int, default=3)
    p.add_argument('--outlier-method', default='iqr', choices=['none','iqr','quantile'])
    p.add_argument('--lower-quantile', type=float, default=.01)
    p.add_argument('--upper-quantile', type=float, default=.99)
    return p.parse_args()


def read(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def outlier_summary(df, method='iqr', lq=.01, uq=.99):
    id_cols={'Customer ID','Month period','cluster_id'}|set(PRODUCTS)
    num=[c for c in df.select_dtypes(include=[np.number]).columns if c not in id_cols]
    rows=[]
    for c in num:
        s=pd.to_numeric(df[c], errors='coerce').dropna()
        if s.empty: continue
        if method=='none':
            lo,hi=np.nan,np.nan; low=high=0
        elif method=='quantile':
            lo,hi=s.quantile(lq),s.quantile(uq); low=int((s<lo).sum()); high=int((s>hi).sum())
        else:
            q1,q3=s.quantile(.25),s.quantile(.75); iqr=q3-q1; lo,hi=q1-1.5*iqr,q3+1.5*iqr; low=int((s<lo).sum()); high=int((s>hi).sum())
        rows.append({'feature':c,'method':method,'lower_cap':lo,'upper_cap':hi,'values_capped_low':low,'values_capped_high':high,'total_outlier_values':low+high,'min_before':s.min(),'p25_before':s.quantile(.25),'median_before':s.median(),'p75_before':s.quantile(.75),'max_before':s.max()})
    return pd.DataFrame(rows).sort_values('total_outlier_values', ascending=False)


def cluster_exact_values(clusters):
    selected_numeric=['Age','Monthly income proxy (sn)','Risk profile score','Days since last login (sn)','Campaign exposure count 90D (sn)','Products ignored count 90D (sn)']
    selected_numeric += [c for c in clusters.columns if c.startswith('digital_') or c.startswith('txn_')]
    selected_numeric=[c for c in selected_numeric if c in clusters.columns]
    selected_cats=[c for c in ['Persona','Lifestage','Income band (sn)','Tier Map','Occupation','Preferred channel (sn)','Gender'] if c in clusters.columns]
    rows=[]
    for cid,g in clusters.groupby('cluster_id'):
        for p in PRODUCTS:
            if p in g.columns:
                s=pd.to_numeric(g[p], errors='coerce')
                rows.append({'cluster_id':cid,'feature_type':'product_holding','feature':p,'mean':s.mean(),'min':s.min(),'p25':s.quantile(.25),'median':s.median(),'p75':s.quantile(.75),'max':s.max(),'top_value':'','top_value_share':np.nan})
        for c in selected_numeric:
            s=pd.to_numeric(g[c], errors='coerce')
            rows.append({'cluster_id':cid,'feature_type':'numeric','feature':c,'mean':s.mean(),'min':s.min(),'p25':s.quantile(.25),'median':s.median(),'p75':s.quantile(.75),'max':s.max(),'top_value':'','top_value_share':np.nan})
        for c in selected_cats:
            ss=g[c].fillna('Missing').astype(str); mode=ss.mode(); top=mode.iloc[0] if not mode.empty else ''
            rows.append({'cluster_id':cid,'feature_type':'categorical','feature':c,'mean':np.nan,'min':np.nan,'p25':np.nan,'median':np.nan,'p75':np.nan,'max':np.nan,'top_value':top,'top_value_share':ss.eq(top).mean() if top else np.nan})
    return pd.DataFrame(rows)


def defining_features(clusters, top_k=10):
    rows=[]
    # products: exact share difference
    for p in PRODUCTS:
        if p in clusters.columns:
            ov=clusters[p].mean()
            for cid,g in clusters.groupby('cluster_id'):
                val=g[p].mean(); diff=val-ov
                rows.append({'cluster_id':cid,'feature_type':'product_holding','feature':f'{p} holding rate','cluster_exact_value':val,'overall_exact_value':ov,'difference':diff,'ranking_difference':abs(diff),'direction':'higher than overall' if diff>0 else 'lower than overall'})
    # numeric: standardize for ranking but keep exact values
    num=['Age','Monthly income proxy (sn)','Risk profile score','Days since last login (sn)','Campaign exposure count 90D (sn)','Products ignored count 90D (sn)']
    num += [c for c in clusters.columns if c.startswith('digital_') or c.startswith('txn_')]
    num=[c for c in num if c in clusters.columns]
    for c in num:
        full=pd.to_numeric(clusters[c], errors='coerce'); ov=full.mean(); sd=full.std() or 1
        for cid,g in clusters.groupby('cluster_id'):
            val=pd.to_numeric(g[c], errors='coerce').mean(); diff=val-ov
            rows.append({'cluster_id':cid,'feature_type':'numeric','feature':c,'cluster_exact_value':val,'overall_exact_value':ov,'difference':diff,'ranking_difference':abs(diff/sd),'direction':'higher than overall' if diff>0 else 'lower than overall'})
    cats=[c for c in ['Persona','Lifestage','Income band (sn)','Tier Map','Occupation','Preferred channel (sn)','Gender'] if c in clusters.columns]
    for c in cats:
        full=clusters[c].fillna('Missing').astype(str)
        for cid,g in clusters.groupby('cluster_id'):
            ss=g[c].fillna('Missing').astype(str); mode=ss.mode(); top=mode.iloc[0] if not mode.empty else ''
            if not top: continue
            cs=ss.eq(top).mean(); os=full.eq(top).mean(); diff=cs-os
            rows.append({'cluster_id':cid,'feature_type':'categorical','feature':f'{c} = {top}','cluster_exact_value':cs,'overall_exact_value':os,'difference':diff,'ranking_difference':abs(diff),'direction':'more common than overall' if diff>0 else 'less common than overall'})
    df=pd.DataFrame(rows)
    if df.empty: return df, df
    df=df.sort_values(['cluster_id','ranking_difference'], ascending=[True,False])
    return df, df.groupby('cluster_id', group_keys=False).head(top_k)


def eval_metrics(top_recs, raw_dir, top_n=3):
    conv=read(Path(raw_dir)/'conversion_data_july_2026.csv')
    if conv.empty or 'Product converted' not in conv.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    conv=conv[['Customer ID','Product converted']].drop_duplicates()
    joined=conv.merge(top_recs[['Customer ID','product_code','recommendation_rank']], left_on=['Customer ID','Product converted'], right_on=['Customer ID','product_code'], how='left')
    rows=[]
    for k in range(1, top_n+1):
        rows.append({'model':'Cluster-First kNN','metric':f'cumulative_hit_rate_at_{k}','value':joined['recommendation_rank'].le(k).sum()/len(conv),'evaluated_conversion_rows':len(conv),'converted_customers':conv['Customer ID'].nunique()})
    ranks=joined['recommendation_rank'].dropna()
    rows.append({'model':'Cluster-First kNN','metric':'mrr','value':(1/ranks).sum()/len(conv) if len(ranks) else 0,'evaluated_conversion_rows':len(conv),'converted_customers':conv['Customer ID'].nunique()})
    long=pd.DataFrame(rows)
    wide={r['metric']:r['value'] for r in rows}; wide.update({'model':'Cluster-First kNN','evaluated_conversion_rows':len(conv),'converted_customers':conv['Customer ID'].nunique()})
    wide=pd.DataFrame([wide])
    rank1=top_recs[top_recs['recommendation_rank']==1][['Customer ID','product_code']].rename(columns={'product_code':'predicted_rank1_product'})
    cb=conv.merge(rank1,on='Customer ID', how='left'); cb['predicted_rank1_product']=cb['predicted_rank1_product'].fillna('NO_RECOMMENDATION')
    labels=PRODUCTS+['NO_RECOMMENDATION']
    cm=confusion_matrix(cb['Product converted'], cb['predicted_rank1_product'], labels=labels)
    cm_df=pd.DataFrame(cm, index=[f'actual_{x}' for x in labels], columns=[f'predicted_{x}' for x in labels]).reset_index().rename(columns={'index':'actual_product'})
    ph=joined.groupby('Product converted', as_index=False).agg(conversions=('Customer ID','count'), top3_hits=('product_code', lambda s:s.notna().sum()))
    ph['top3_hit_rate']=ph['top3_hits']/ph['conversions']
    return long,wide,cm_df,ph


def make_plots(output_dir, dataset, clusters, profiles, top_defs, top_recs, hit_wide):
    out=Path(output_dir); viz=out/'visualizations'; viz.mkdir(exist_ok=True)
    # defining features per cluster
    for cid,g in top_defs.groupby('cluster_id'):
        plot=g.head(6).sort_values('difference')
        plt.figure(figsize=(9,4)); plt.barh(plot['feature'].astype(str), plot['difference']); plt.axvline(0,lw=1)
        plt.xlabel('Exact difference from overall average/share'); plt.title(f'{dataset}: Cluster {cid} defining features')
        plt.tight_layout(); plt.savefig(viz/f'08_cluster_{cid}_top_defining_features_exact.png',dpi=150,bbox_inches='tight'); plt.close()
    # age range
    if 'Age' in clusters.columns:
        r=[]
        for cid,g in clusters.groupby('cluster_id'):
            s=pd.to_numeric(g['Age'],errors='coerce'); r.append({'cluster_id':cid,'min':s.min(),'median':s.median(),'max':s.max()})
        rr=pd.DataFrame(r); rr.to_csv(out/'report_age_range_by_cluster.csv',index=False)
        plt.figure(figsize=(8,4)); plt.plot(rr['cluster_id'].astype(str),rr['min'],marker='o',label='Min'); plt.plot(rr['cluster_id'].astype(str),rr['median'],marker='o',label='Median'); plt.plot(rr['cluster_id'].astype(str),rr['max'],marker='o',label='Max')
        plt.xlabel('Cluster ID'); plt.ylabel('Age'); plt.title(f'{dataset}: age range by cluster'); plt.legend(); plt.tight_layout(); plt.savefig(viz/'09_age_range_by_cluster.png',dpi=150,bbox_inches='tight'); plt.close()
    # top1 mix
    if not top_recs.empty:
        rank1=top_recs[top_recs['recommendation_rank']==1]
        counts=rank1.groupby(['cluster_id','product_code']).size().unstack(fill_value=0).reindex(columns=PRODUCTS, fill_value=0)
        counts.to_csv(out/'report_top1_recommendation_counts_by_cluster.csv')
        ax=counts.plot(kind='bar',stacked=True,figsize=(9,5)); ax.set_xlabel('Cluster ID'); ax.set_ylabel('Top-1 recommendation count'); ax.set_title(f'{dataset}: top-1 recommendation counts by cluster and product'); plt.tight_layout(); plt.savefig(viz/'10_top1_recommendation_counts_by_cluster.png',dpi=150,bbox_inches='tight'); plt.close()
    # cumulative hit
    if not hit_wide.empty and 'cumulative_hit_rate_at_3' in hit_wide.columns:
        cols=[c for c in ['cumulative_hit_rate_at_1','cumulative_hit_rate_at_2','cumulative_hit_rate_at_3'] if c in hit_wide.columns]
        plt.figure(figsize=(7,4)); plt.bar([c.replace('cumulative_hit_rate_at_','Hit@') for c in cols],[hit_wide[c].iloc[0] for c in cols]); plt.ylabel('Cumulative hit rate'); plt.title(f'{dataset}: cumulative top-3 hit rates'); plt.tight_layout(); plt.savefig(viz/'06_cumulative_hit_rates.png',dpi=150,bbox_inches='tight'); plt.close()


def main():
    a=parse_args(); raw=Path(a.raw_dir); out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True); dataset=a.dataset_name or raw.name
    clusters=read(out/'cluster_first_customer_clusters.csv'); top_recs=read(out/'final_top_3_recommendations_cluster_first_knn.csv'); profiles=read(out/'cluster_first_cluster_profiles.csv')
    if clusters.empty or top_recs.empty:
        raise FileNotFoundError('Run cluster_first_knn_recommender.py first; required outputs missing.')
    exact=cluster_exact_values(clusters); exact.to_csv(out/'cluster_exact_feature_values.csv', index=False)
    all_defs, top_defs=defining_features(clusters, top_k=10); all_defs.to_csv(out/'cluster_defining_features_all_exact_values.csv', index=False); top_defs.to_csv(out/'cluster_defining_features_top10_exact_values.csv', index=False)
    outliers=outlier_summary(clusters, a.outlier_method, a.lower_quantile, a.upper_quantile); outliers.to_csv(out/'outlier_treatment_summary.csv', index=False)
    hit_long,hit_wide,cm,prod_hit=eval_metrics(top_recs, raw, a.top_n)
    hit_long.to_csv(out/'cumulative_hit_rate_metrics.csv', index=False); hit_wide.to_csv(out/'cluster_first_knn_accuracy_against_july_conversions.csv', index=False); cm.to_csv(out/'confusion_matrix_rank1_actual_vs_predicted.csv', index=False); prod_hit.to_csv(out/'top3_hit_rate_by_actual_product.csv', index=False)
    make_plots(out,dataset,clusters,profiles,top_defs,top_recs,hit_wide)
    # compact run summary
    summary={'dataset':dataset,'customers':clusters['Customer ID'].nunique(),'clusters':clusters['cluster_id'].nunique(),'recommendation_rows':len(top_recs),'outlier_method':a.outlier_method}
    if not hit_wide.empty:
        for c in hit_wide.columns:
            if c!='model': summary[c]=hit_wide[c].iloc[0]
    pd.DataFrame([summary]).to_csv(out/'dataset_run_summary.csv', index=False)
    print('Postprocess complete:', out)

if __name__=='__main__': main()
