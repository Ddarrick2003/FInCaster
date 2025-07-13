from arch import arch_model

def forecast_garch_var(df):
    returns = df['Returns'] * 100  # scaled returns for GARCH
    am = arch_model(returns, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    forecast = res.forecast(horizon=5)
    vol_forecast = forecast.variance.iloc[-1] ** 0.5
    var_1d = 1.65 * vol_forecast.iloc[0]  # 95% VaR
    return vol_forecast, var_1d
