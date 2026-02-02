import numpy as np
from scipy import integrate
import csv
import pandas as pd

def regress(x):
    """
    定义被积函数 y = -0.0118x^2 + 0.5762x + 11.7948
    """
    return -0.0118 * x**2 + 0.5762 * x + 11.7948

def calculate(age):
    """
    计算从a到age的定积分
    
    参数:
        a: 积分下限
        age: 积分上限
    
    返回:
        积分值T
    """
    result = regress(age+1)/regress(age)
    return result

def main():
    """
    主函数：计算AGE从19到39的积分值并保存到CSV
    
    参数:
        a: 积分下限（默认为0，可修改）
    """
    # 存储结果
    results = []
    
    # 计算AGE从19到39的积分
    for age in range(15, 50):
        Norm_T = calculate(age)
        results.append({'AGE' : age, 'T' : Norm_T})
        print(f"AGE = {age}, T = {Norm_T:.6f}")
    
    # 保存到CSV文件
    output_file = 'csv_fold/potential_results.csv'# 写入表头
    result_df=pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()

