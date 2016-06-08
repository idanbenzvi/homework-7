import weka.core.Instance;
import weka.core.Instances;

import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Random;
import java.util.Set;

public class KMeans {



    public int K = 2;
    private final int NUM_ITERATIONS = 500;
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

        //randomly select K values in the range of our instances (in order to select K random instances as centroids)
        Set<Integer> generated = new LinkedHashSet<Integer>();
        while (generated.size() < K)
        {
            Integer next = rand.nextInt(instances.numInstances()) ;
            // As we're adding to a set, this will automatically do a containment check - no duplicates ! :-)
            generated.add(next);
        }

        //create an empty instances object with capacity K
        Instances centroidInstances = new Instances(instances, K);

        //create an iterator and then assign centroid instances with the random generated values
        Iterator randInstances = generated.iterator();

        while(randInstances.hasNext()){
            centroidInstances.add(instances.instance((int) randInstances.next()));
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
            double[][] centroidMeans = new double[K][4];

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

        //calculate the square root of
        //the sum of squared distances of every instance from the closest centroid to it
        double addSum = 0;

        for (int i = 0; i < instances.numInstances() ; i++){
            addSum += calcSquaredDistance(instances.instance(i),centroidInstances.instance(centroidAlloc[i]));
        }

        addSum = Math.sqrt(addSum);

        return addSum;
    }

    /**
     * a simple setter for our K number for the model (num of centroids)
     * @param k
     */
    public void setK(int k) {
        K = k;
    }

    public int[] getCentroidAlloc() {
        return centroidAlloc;
    }

    public Instances getCentroidInstances() {
        return centroidInstances;
    }

    public void setCentroidAlloc(int[] centroidAlloc) {
        this.centroidAlloc = centroidAlloc;
    }

    public void setCentroidInstances(Instances centroidInstances) {
        this.centroidInstances = centroidInstances;
    }
}
