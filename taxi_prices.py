# This script runs the application on a local server.
# It contains the definition of routes and views for the application.

import flask
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#---------- MODEL IN MEMORY ----------------#

# setting x and y
df = pd.read_csv('taxi.csv')

# df = df.drop(['pickup_time'],1)
X = df[['pickup_lon','pickup_lat','hour']]
y = df['total_amount']

PREDICTOR = RandomForestRegressor().fit(X,y)

#---------- CREATING AN API, METHOD 1 ----------------#

# Initialize the app
app = flask.Flask(__name__)


# When you navigate to the page 'server/predict', this will run
# the predict() function on the parameters in the url.
#
# Example URL:
# http://localhost:4000/predict?pickup_lon=-73.475623&pickup_lat=40.772377&hour=18
@app.route('/predict', methods=["GET"])
def predict():
    '''Makes a prediction'''
    pickup_lon = float(flask.request.args['pickup_lon'])
    pickup_lat = float(flask.request.args['pickup_lat'])
    hour = float(flask.request.args['hour'])

    item = np.array([pickup_lon, pickup_lat, hour])
    item = item.reshape(1, -1)
    score = PREDICTOR.predict(item)
    results = {'price': round(score[0],2)}
    return flask.jsonify(results)

#---------- CREATING AN API, METHOD 2 ----------------#

#
if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = '4000'

    app.run(HOST, PORT)
