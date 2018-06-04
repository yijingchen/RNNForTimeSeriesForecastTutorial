def mape(predictions, actuals):
    return ((predictions - actuals).abs() / actuals).mean()