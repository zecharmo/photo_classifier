
# Use k-NN and Naive Bayes to classify images as either landscapes or portraits

# Start with collecting data 
# Grab 2000 pictures from flickr and ask for photos that have already been tagged
r = requests.get('flickr.photos.search(api_key=key, tags=landscape,portrait, count=2000)')
df = json_normalize(r.json()['photoSearch'])

# create database and connect
con = lite.connect('photos.db')
cur = con.cursor()

# create a table to store information about each photo
with con:
    cur.execute('DROP TABLE IF EXISTS photo_grams')
    cur.execute('CREATE TABLE photo_grams (id INT PRIMARY KEY, redbin1 INT, redbin2 INT, redbin3 INT, redbin4 INT, redbin5 INT, 
															   greenbin1 , greenbin2 INT, greenbin3 INT, greenbin4 INT, greenbin5 INT,
															   bluebin1 , bluebin2 INT, bluebin3 INT, bluebin INT, bluebin5 INT)')
    
	# Represent each image with a binned RGB (red, green, blue) intensity histogram
	# For each of red, green, and blue, measure the intensity, which is a number between 0 and 255
	# Draw three binned histograms with counts of the number of pixels of intensity 0-50, 51-100, 101-150, 151-200, 201-255
	for photo in r.json()['photoSearch']
		red_gram = numpy.histogram(photo, bins=[0,51,101,151,201])
		green_gram = numpy.histogram(photo, bins=[0,51,101,151,201])
		blue_gram = numpy.histogram(photo, bins=[0,51,101,151,201])
        
		# update table with info on histograms
		cur.execute(sql,(photo['id'], photo['redbin1'], photo['redbin2'], photo['redbin3'], photo['redbin4'], photo['redbin5'], 
									  photo['greenbin1'], photo['greenbin2'], photo['greenbin3'], photo['greenbin4'], photo['greenbin5'],
									  photo['bluebin1'], photo['bluebin2'], photo['bluebin3'], photo['bluebin4'], photo['bluebin5'],))
		
# We now have 15 numbers, corresponding to 3 colors and 5 bins per color

# Use k-NN to decide how much “blue” makes a landscape versus a portrait 
# create training data set
dfTrain, dfTest = train_test_split(df, test_size=0.5)

# this is the subset of labels for the training set
cl = dfTrain[:,5]
# subset of labels for the test set, we're withholding these
true_labels  dfTest[:,5]

# we'll loop through and see what the misclassification rate is for different values of k
for k in range(1,10):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(dfTrain[:,:5], dfTrain[:,5])
    # make predictions
    expected = dfTest[:,5]
    predicted = model.predict(dfTest[:,:5])
    # misclassification rate
    error_rate = (predicted != expected).mean()
    print('%d:, %.2f' % (k, error_rate))

# apply k Nearest Neighbors with the value where k has the lowest misclassification rate (written below as k)
model = KNeighborsClassifier(n_neighbors=k)
model.fit(dfTrain[:,:5], dfTrain[:,5])

# visualize results from k-NN
# see the default parameters below; you can initialize with none, any, or all of these.
model = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1)

# apply your knowledge of Naive Bayes in this problem
# landscape below refers to number of photos tagged as landscapes out of the total number of photos
# we want to take the average amount of blue in each bin accross all the training data
 P(landscape|blue) = ((P(bluebin1|landscape) + P(bluebin2|landscape) + P(bluebin3|landscape) + P(bluebin4|landscape) + P(bluebin5|landscape))*P(landscape))/
					   ((AVG(bluebin1) + (AVG(bluebin2) + (AVG(bluebin3) + (AVG(bluebin4) + (AVG(bluebin5))


