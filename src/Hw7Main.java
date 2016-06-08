import weka.core.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;

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

    }

}
