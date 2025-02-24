from modules import FirstFit, BestFit, CombinationAlgorithm
import pandas as pd

# stocks = [(150, 150), (120, 120), (100, 100)]
# products = [(130, 130), (150, 50), (140, 40), (140, 60), (10, 100), (25, 25),(140, 60)]
def LOGIC(df_stock, df_required, method_option):
    
    stocks = []
    for i in range(1, len(df_stock) + 1):
        row = df_stock.loc[i]  
        stocks.extend([(row["Width"], row["Length"])] * row["Quantity"]) 
        
    products = []
    for i in range(1, len(df_required) + 1):
        row = df_required.loc[i]  
        products.extend([(row["Width"], row["Length"])] * row["Quantity"]) 
    
    if method_option == "First Fit":
        cutting_problem = FirstFit.CuttingStockProblem2D(stocks, products)
        cutting_problem.solve()
        images = cutting_problem.visualize()
    
    if method_option == "Best Fit":
        best_fit_cutting = BestFit.BestFitCuttingStock(stocks, products)
        best_fit_cutting.solve()
        images = best_fit_cutting.visualize()
        
    if method_option == "Combination Algorithm":
        combination_cutting = CombinationAlgorithm.CombinationCuttingStock(stocks, products)
        combination_cutting.solve()
        images = combination_cutting.visualize()


    return images


