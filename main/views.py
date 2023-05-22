from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import jinja2

def index(request):
    return render(request, 'main/index.html')

def visual(request):
    # получаем из данных запроса POST отправленные через форму данные
    int(request.POST.get("n_clusters", 2))
    adresses = request.POST.getlist("adress") ##адреса строками
    start_point = request.POST.get("start")

    print(adresses)
    geolocator = Nominatim(user_agent="Tester")
    points = np.zeros((len(adresses), 2))
    x_points = np.zeros(len(adresses))
    y_points = np.zeros(len(adresses))
    for i in range(len(adresses)):
        adress = adresses[i]
        print(adress)
        location = geolocator.geocode(adress)

        points[i][0] = location.latitude
        points[i][1] = location.longitude
        x_points[i] = float(location.latitude)
        y_points[i] = float(location.longitude)
    print(points, x_points, y_points)
    n_clusters = int(request.POST.get("n_clusters", 2))

    #points = np.array([[6,7], [7, 8], [8, 7], [9, 5]])
    kmeans = KMeans(n_clusters)
    kmeans.fit(points)
    start = np.array([54, 43])
    clusters = [[] for _ in range(n_clusters)]
    adress_in_cluster = [[] for _ in range(n_clusters)]
    for i in range(len(points)):
        clusters[kmeans.labels_[i]].append([x_points[i], y_points[i]])
        adress_in_cluster[kmeans.labels_[i]].append([adresses[i]])

        ##массив точек
    print('add', adress_in_cluster)
    print('cl',clusters)
    string = ""
    way_in_cluster = []
    for i in range(n_clusters):

        best_way = ant_colony_optimization(np.array(clusters[i]), n_ants=n_clusters, n_iterations=100, alpha=1, beta=1, evaporation_rate=0.5, Q=1)
        way_in_cluster.append(best_way)
    print(way_in_cluster)
    for i in range(len(way_in_cluster)):
        numb = i
        string += "<br>"+"Для транспортного средства " + str(numb+1) + " маршрут выглядит следующим образом: " + start_point + "->"
        for j in range(len(way_in_cluster[i])):
            k = way_in_cluster[i][j]
            string+= adress_in_cluster[i][k][0] + "->"
        string += start_point + "\n"
        print(string)
    return render(request, 'main/visual.html', {"number_of_clusters": n_clusters, "labels": list(kmeans.labels_), "adresses": adresses, "points_x": x_points.tolist(), "points_y": y_points.tolist(), "clusters": clusters, "result": string})


def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def ant_colony_optimization(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf

    for iteration in range(n_iterations):
        paths = []
        path_lengths = []

        for ant in range(n_ants):
            visited = [False] * n_points
            current_point = np.random.randint(n_points)
            visited[current_point] = True
            path = [current_point]
            path_length = 0

            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))

                for i, unvisited_point in enumerate(unvisited):
                    probabilities[i] = pheromone[current_point, unvisited_point] ** alpha / distance(
                        points[current_point], points[unvisited_point]) ** beta

                probabilities /= np.sum(probabilities)

                next_point = np.random.choice(unvisited, p=probabilities)
                path.append(next_point)
                path_length += distance(points[current_point], points[next_point])
                visited[next_point] = True
                current_point = next_point

            paths.append(path)
            path_lengths.append(path_length)

            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

        pheromone *= evaporation_rate

        for path, path_length in zip(paths, path_lengths):
            for i in range(n_points - 1):
                pheromone[path[i], path[i + 1]] += Q / path_length
            pheromone[path[-1], path[0]] += Q / path_length

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(points[:, 0], points[:, 1], c='r', marker='o')

    for i in range(n_points - 1):
        print([points[best_path[i], 0]], [points[best_path[i], 1]])
        ax.plot([points[best_path[i], 0], points[best_path[i + 1], 0]],
                [points[best_path[i], 1], points[best_path[i + 1], 1]],
                c='g', linestyle='-', linewidth=2, marker='o')

    ax.plot([points[best_path[0], 0], points[best_path[-1], 0]],
            [points[best_path[0], 1], points[best_path[-1], 1]],
            c='g', linestyle='-', linewidth=2, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.show()
    return best_path
# Create your views here.
