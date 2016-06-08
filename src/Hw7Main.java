import weka.core.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

//import weka.filters.unsupervised.attribute.PrincipalComponents;

public class Hw7Main {
    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    public static Instances convertImgToInstances(BufferedImage image) {
        Attribute attribute1 = new Attribute("alpha");
        Attribute attribute2 = new Attribute("red");
        Attribute attribute3 = new Attribute("green");
        Attribute attribute4 = new Attribute("blue");
        FastVector fvWekaAttributes = new FastVector(4);
        fvWekaAttributes.addElement(attribute1);
        fvWekaAttributes.addElement(attribute2);
        fvWekaAttributes.addElement(attribute3);
        fvWekaAttributes.addElement(attribute4);
        Instances imageInstances = new Instances("Image", fvWekaAttributes, image.getHeight() * image.getWidth());

        int[][] result = new int[image.getHeight()][image.getWidth()];
        int[][][] resultARGB = new int[image.getHeight()][image.getWidth()][4];

        for (int col = 0; col < image.getWidth(); col++) {
            for (int row = 0; row < image.getHeight(); row++) {
                int pixel = image.getRGB(col, row);

                int alpha = (pixel >> 24) & 0xff;
                int red = (pixel >> 16) & 0xff;
                int green = (pixel >> 8) & 0xff;
                int blue = (pixel) & 0xff;
                result[row][col] = pixel;
                resultARGB[row][col][0] = alpha;
                resultARGB[row][col][1] = red;
                resultARGB[row][col][2] = green;
                resultARGB[row][col][3] = blue;

                Instance iExample = new DenseInstance(4);
                iExample.setValue((Attribute) fvWekaAttributes.elementAt(0), alpha);// alpha
                iExample.setValue((Attribute) fvWekaAttributes.elementAt(1), red);// red
                iExample.setValue((Attribute) fvWekaAttributes.elementAt(2), green);// green
                iExample.setValue((Attribute) fvWekaAttributes.elementAt(3), blue);// blue
                imageInstances.add(iExample);
            }
        }

        return imageInstances;

    }


    public static BufferedImage convertInstancesToImg(Instances instancesImage, int width, int height) {
        final BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int index = 0;
        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                Instance instancePixel = instancesImage.instance(index);
                int pixel = ((int) instancePixel.value(0) << 24) | (int) instancePixel.value(1) << 16
                        | (int) instancePixel.value(2) << 8 | (int) instancePixel.value(3);
                image.setRGB(col, row, pixel);
                index++;
            }
        }
        return image;
    }

    public static void main(String[] args) throws Exception {
        //create instances object from image and quantize it using Kmeans
        BufferedImage image = ImageIO.read(new File("baboon_face.jpg"));

        //convert image to instances
        Instances imageInstances = convertImgToInstances(image);

        //run K-menas algorithm on the selected image file
        KMeans KmeansClassifier = new KMeans();
        KmeansClassifier.buildClusterModel(imageInstances);

        Instances quantizedInstances = KmeansClassifier.quantize(imageInstances);
        BufferedImage out = convertInstancesToImg(quantizedInstances,image.getWidth(),image.getHeight());

        File outputfile = new File("output.jpg");
         ImageIO.write(out, "jpg", outputfile);


        //run PCA looping over number of principal components and print the average
        // distance of transformed instances from original instances
        //run PCA from i = 13 .. 90

        //read Libras dataset into an instances object
        Instances libras = loadData("libras.txt");

        //init. an arraylist later to be exported as csv file
        ArrayList<Double> pcaResult = new ArrayList<Double>();

        PrincipalComponents pca = new PrincipalComponents();
        for(int i = 13 ; i < libras.numAttributes() ; i++) {
            //calc the average distance of the instances from the original instances after the PCA performed over the original dataset has been transformed
            pca.setNumPrinComponents(i);
            pca.setTransformBackToOriginal(true);
            pca.buildEvaluator(libras);
            Instances data2 = pca.transformedData(libras);
            double dist = calcAvgDistance(data2, libras);

            //output resulting distance to the systemout and retain results in a datastructuer later to be written to a file to be used
            //fro a an excel spreadsheet scatterplot
            System.out.println(dist);
            pcaResult.add(dist);
        }

        //output the results from the array list to a csv file
        writeToCSV(pcaResult,"PCAOutput.csv");

    }

//    After running all of the necessary steps of the KMeans section we will run the PCA algorithm on the libras.txt data set. You should do this in the main method.
//    We will run PCA many times, each time using a different number of principal components. So loop through
//            (notice that the libras data set has 91 features initially, so using 91 principal components should be the exact recovery of the data)
//    and do the following: run PCA on the instances and transform them back to the original space using principal components,
//    measure the average Euclidean distance of the new data set from the original data set and print this average distance. An explanation of how to use the PCA algorithm provided is below under technical hints.
//

    /**
     * calculate the euclidean distance between all instances of the transformed set and the original set of the PCA Libra
     * datset
     * @param instances
     * @param originalSet
     */
    public static double calcAvgDistance(Instances instances,Instances originalSet){
        //calculate the euclidean distance between every 2 corresponding instances

        double distance = 0;

        //iterate over all instances from the dataset
        for (int i = 0 ; i < instances.numInstances() ; i++) {
            double dist = 0;

            //calculate the distance
            for (int j = 0; j < originalSet.numAttributes(); j++) {
                dist += Math.pow((instances.instance(i).value(j) - originalSet.instance(i).value(j)), 2);
            }

            //calculate the sqare root of the distance
            distance += Math.sqrt(dist);
        }

        //calculate the mean by dividing the above result by the number of instances
        distance /= instances.numInstances();

    return distance;
    }

    //*** FILE HANDLING METHODS FOR DATASET LOADING ***


    /**
     * Load the data from the inputreader into an instances object
     * @param fileName
     * @return Instances data
     * @throws IOException
     */
    public static Instances loadData(String fileName) throws IOException {
        BufferedReader datafile = readDataFile(fileName);
        Instances data = new Instances(datafile);
        return data;
    }

    /**
     * function that outputs the arraylist with the PCA distances for 13 .. 90 principal components to a CSV
     * @param csvInput
     * @param fileName
     * @throws IOException
     */
    private static void writeToCSV(ArrayList<Double> csvInput,String fileName) throws IOException{

        FileWriter writer = new FileWriter(fileName);

        //heading for the first column and the second column
        writer.append("principal components");
        writer.append(',');
        writer.append("Average Distance");
        writer.append('\n');

        //first principle component number
        int pnum = 13;

        //go over all principal component distances as calculated by the PCA model and output them to CSV
        for(int i = 0 ; i < csvInput.size(); i++){
            writer.append(Integer.toString(pnum++));
            writer.append(',');
            writer.append(Double.toString(csvInput.get(i)));
            writer.append(',');
            writer.append('\n');
        }

        //close and cleanup
        writer.flush();
        writer.close();

    }
}
