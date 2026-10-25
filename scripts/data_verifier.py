from src.data_loader import load_data
import pandas as pd
df=load_data()
labeled=df["failureType"].apply(lambda x: x[0][0] if len(x) > 0 else None)
print("Starting data verification from WM-811K dataset\n")
print(f'No. of data rows = 811,457 \tVerified = {df.shape[0]==811457}')
print(f"No. of labeled rows = 172,950\tVerified = {(df['failureType'].apply(len) > 0).sum()==172950}")
print(f"No. of defect rows = 25,519\tVerified = {((labeled.notna()) & (labeled!="none")).sum()==25519}")
print(f"No. of None rows = 147,431\tVerified = {((labeled.notna()) & (labeled=="none")).sum()==147431}")
if((df.shape[0]==811457 and (df['failureType'].apply(len) > 0).sum()==172950 and ((labeled.notna()) & (labeled!="none")).sum()==25519 and ((labeled.notna()) & (labeled=="none")).sum()==147431)):
    print("\nAll data checks passed")
else:
    raise SystemExit(f"\nData verification failed")
