def haversine(lon1, lat1, lon2, lat2):
    # Calculate distances between lat/lon in miles.
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    newlon = lon2 - lon1
    newlat = lat2 - lat1

    haver_formula = np.sin(newlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon / 2.0) ** 2

    dist = 2 * np.arcsin(np.sqrt(haver_formula))
    dist = 3958 * dist  # 6367 for distance in KM for miles use 3958
    return dist

