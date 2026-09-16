import numpy as np
FEED_COLUMNS=['feed_MeOH','feed_EtAC','feed_Water','feed_flow','feed_pressure','feed_temperature']
def nearest(df,feed,n=10):
    a=df[FEED_COLUMNS].to_numpy(float); lo=a.min(0); scale=np.maximum(a.max(0)-lo,1e-9); q=np.array([feed[c] for c in FEED_COLUMNS]); d=np.linalg.norm((a-q)/scale,axis=1); return df.assign(feed_distance=d).nsmallest(n,'feed_distance')
