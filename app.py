from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import json
import plotly
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    # Get user inputs
    ticker = request.form.get('ticker')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    forecast_days = int(request.form.get('forecast_days', 365))
    
    try:
        # Download the data
        data = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False, auto_adjust=False)
        
        if data.empty:
            return jsonify({"error": "No data found for the specified ticker and date range."})
        
        # Prepare the data for Prophet
        data.reset_index(inplace=True)
        data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        df = data[['ds', 'y']].copy()
        df = df.dropna()
        
        # Train the model
        m = Prophet()
        m.fit(df)
        
        # Make future predictions
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)
        
        # Create the plot
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=df['ds'],
            y=df['y'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        # Add forecast intervals
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(width=0),
            name='Confidence Interval'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Stock Forecast for {ticker}',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white'
        )
        
        # Convert the plot to JSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Prepare forecast data for table display
        forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10).copy()
        forecast_table['ds'] = forecast_table['ds'].dt.strftime('%Y-%m-%d')
        forecast_table = forecast_table.round(2).to_dict('records')
        
        return jsonify({
            "graph": graphJSON, 
            "forecast_data": forecast_table,
            "success": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/components')
def components():
    # For demonstration purposes, show components plots
    ticker = request.args.get('ticker', '9988.HK')
    start_date = request.args.get('start_date', '2022-01-01')
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    try:
        # Download the data
        data = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False, auto_adjust=False)
        data.reset_index(inplace=True)
        data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        df = data[['ds', 'y']].copy()
        df = df.dropna()
        
        # Train the model
        m = Prophet()
        m.fit(df)
        
        # Generate the components plot
        fig_comp = m.plot_components(m.predict(m.make_future_dataframe(periods=365)))
        
        # Convert to plotly figure
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode('utf-8')
        components_image = f'data:image/png;base64,{string}'
        plt.close()
        
        return render_template('components.html', components_image=components_image, ticker=ticker)
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
