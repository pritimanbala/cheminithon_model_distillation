class RetrainingPolicy:
 def __init__(self,every_n=100,shift_threshold=.2): self.every_n,self.shift_threshold=every_n,shift_threshold
 def should_retrain(self,new_records,feed_shift=0,performance_drop=False,manual=False): return manual or performance_drop or new_records>=self.every_n or feed_shift>=self.shift_threshold
