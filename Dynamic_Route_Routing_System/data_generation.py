import random
import pandas as pd

# Simulate relevant bus routing data
def generate_traffic_data(num_routes=20, num_samples=500):
    routes = ['Route_' + str(i) for i in range(1, num_routes + 1)]
    data = []

    for route in routes:
        for i in range(num_samples):
            traffic_condition = random.choice(['low', 'medium', 'high'])
            road_condition = random.choice(['good', 'average', 'poor'])
            weather_condition = random.choice(['sunny', 'rainy', 'foggy'])
            bus_capacity = random.randint(10, 50)  # Number of passengers
            avg_speed = random.uniform(20, 60)  # Average speed in km/h
            distance_to_destination = random.uniform(5, 30)  # Distance in km
            time_of_day = random.choice(['morning', 'afternoon', 'evening', 'night'])
            day_type = random.choice(['weekday', 'weekend'])

            # Calculate delay
            base_delay = random.uniform(5, 20)  # Base delay in minutes

            if traffic_condition == 'high':
                base_delay += random.uniform(10, 20)
            if road_condition == 'poor':
                base_delay += random.uniform(5, 15)
            if weather_condition in ['rainy', 'foggy']:
                base_delay += random.uniform(5, 10)
            if time_of_day in ['morning', 'evening']:  # Rush hours
                base_delay += random.uniform(5, 15)
            if day_type == 'weekend':
                base_delay -= random.uniform(0, 5)  # Less traffic on weekends

            data.append([
                route, traffic_condition, road_condition, weather_condition, bus_capacity, avg_speed,
                distance_to_destination, time_of_day, day_type, base_delay
            ])

    df = pd.DataFrame(data, columns=[
        'Route', 'TrafficCondition', 'RoadCondition', 'WeatherCondition', 'BusCapacity', 'AvgSpeed',
        'DistanceToDestination', 'TimeOfDay', 'DayType', 'Delay'
    ])
    return df

# Generate dataset
traffic_data = generate_traffic_data()
traffic_data.to_csv('data/traffic_data.csv', index=False)
print("Data generated successfully!")
