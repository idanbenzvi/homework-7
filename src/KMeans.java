import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

public class KMeans {

    private final int K = 8;
    private final int NUM_ITERATIONS = 100;
    private double[][] centroids = new double[K][4]; // data structure to represent the centroids
    private int centroidAlloc[];
    private Instances centroidInstances;

    /**
     * output: should initialize any parameters (therefore should call initializecentroids)
     * and run the K- Means algorithm (which means to call findKMeansCentroids methods).
     * It should run everything necessary to find the K clusters centroids on the input instances.
     *
     * @param instances
     */
    public void buildClusterModel(Instances instances) {
        //initialize parameters
        this.centroidInstances = initializeCentroids(instances);

        //run K-means algorithm
        findKMeansCentroids(instances);
    }


    /**
     * should initialize the centroids by selecting k random instances from the training set and setting the centers as those instances.
     * This method should decide what points the centroids are initially.
     *
     * @param instances
     * @return a group of random K instances that will be used as the centroids of our clusters (initial centroids)
     */
    public Instances initializeCentroids(Instances instances) {
        //randomize the instances in the set in order to select K instances as initial centroids (also better in general)
        Random rand = new Random(12345);
        //shuffles the instances according to the random seed provided above
        instances.randomize(rand);

        Instances centroidInstances = new Instances(instances, 0, K);

        //also store centroids in our array structure for further calculations
        for (int i = 0; i < K; i++) {
            //get a random instance
            Instance currInst = instances.get(i);
            //set centroid values using the selected instance
            for (int j = 0; j < 4; j++) {
                centroids[i][j] = currInst.value(j);
            }
        }

        return centroidInstances;
    }

    /**
     * output: should find/store the centroids according to the KMeans algorithm
     *
     * @param instances
     * @return
     */
    public Instances findKMeansCentroids(Instances instances) {
        //since our centroids are the first k instances, we will avoid using them within the context of the current method

        centroidAlloc = new int[(instances.numInstances())];

        //allocate the instance to the centroid (temp array)

        //loop until set iterations have been completed
        for (int itr = 0; itr < NUM_ITERATIONS; itr++) {

            //reset centroid instance counter variables
            int[] instanceCountCentroid = new int[K];

            //find the closest centroid for each instance and save it in a temporary array
            for (int i = 0; i < instances.numInstances(); i++) {
                centroidAlloc[i] = findClosestCentroid(instances.instance(i));
                //increment the counter for this centroid (of instances)
                instanceCountCentroid[centroidAlloc[i]] += 1;
            }

            //calculate the new mean of the centroid
            int[][] centroidMeans = new int[K][4];

            for (int i = 0; i < instances.numInstances(); i++) {
                for (int j = 0; j < 4; j++) {
                    //for each instance, according to the centroid it was assigned to, add the values from each attribute
                    //to a total sum representing the centroid's mean. Once the outer loop iterates over all the instances
                    //each centroid's attribute will be divided by the number of instances the centroid was associated with
                    //to calculate the mean of the centroid.

                    //NOTE: starting from K is due to the fact the first K instance were used as our centroids
                    centroidMeans[centroidAlloc[i]][j] += instances.instance(i).value(j);
                }
            }

            //now calculate the mean for each centroid (divide by the number of instances associated with it
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < 4; j++) {
                    centroidMeans[i][j] /= instanceCountCentroid[i];
                    //All centroid means have been calculated - replace the centroid instance attribute values according to the new
                    //means calculated: each instance is a centroid, the values are the attribute values
                    centroidInstances.instance(i).setValue(j, centroidMeans[i][j]);
                }
            }

        }

        return centroidInstances;
    }

    /**
     * calculate the euclidean distance between 2 instances
     *
     * @param a
     * @param b
     * @return
     */
    public double calcSquaredDistance(Instance a, Instance b) {
        //go over all attributes of our instances and calculate the squared dist(euclidean)
        int dist = 0;

        for (int i = 0; i < a.numAttributes(); i++) {
            dist += Math.pow((a.value(i) - b.value(i)), 2);
        }

        double distance = Math.sqrt(dist);

        return distance;
    }


    /**
     * calculate which centroid is closest to the current instance
     *
     * @return
     */
    public int findClosestCentroid(Instance instance) {
        //calculate distances
        double minDist = Double.MAX_VALUE;
        double curDist = 0;
        int closestIndex = 0;

        for (int i = 0; i < K; i++) {
            curDist = calcSquaredDistance(instance, centroidInstances.instance(i));
            if (curDist < minDist) {
                minDist = curDist;
                closestIndex = i;
            }
        }
        //return the index of the closest centroid
        return closestIndex;
    }

    /**
     * output: should replace every instance in Instances to the centroid to which it is closest to and return the new Instances object.
     *
     * @param instances
     * @return
     */
    public Instances quantize(Instances curinstances) {
        //using the centroid allocation array we've created while creating the centroids, we will replace each instance according to the centroid it is
        //assoicated with (using centroidInstances object we've created)
        for(int i = 0 ; i < curinstances.numInstances() ; i++){
            for(int j = 0 ; j < 4 ; j++) {
                //for each instance - replace it's values with those of the centroid affiliated with it
                //specifically - for the instance i, find what centroid it is associated with via centroid instances field, and the centroid allocated to it
                //then set the requires values
                curinstances.instance(i).setValue(j,centroidInstances.instance(centroidAlloc[i]).value(j));
            }
        }

        return curinstances;
    }


    /**
     * output: should calculate the average within set sum of squared error.
     * That is it should calculate the square root of the sum of the squared distances of every instance from the closest centroid to it.
     * Return the double value of the WSSSE. This is the cost function learned in class.
     */
    public double calcAvgWSSSE(Instances instances) {
        return 0;
    }

}
