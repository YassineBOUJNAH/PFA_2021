package org.example;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.HashMap;
import java.util.List;

public class HeartDiseasesPredictionRF {
    public static void main(String[] args) {
        //Create an active Spark session
        SparkSession spark = UtilityForSparkSession.mySession();

        // Taken input and create the RDD from the dataset by specifying the  input source and number of partition. Adjust the number of partition basd on your dataser size
        long model_building_start = System.currentTimeMillis();
        String input = "src/main/java/org/example/heart_diseases/processed_cleveland.data";
        //String new_data = "heart_diseases/processed_hungarian.data";
        RDD<String> linesRDD = spark.sparkContext().textFile(input, 2);

        JavaRDD<LabeledPoint> data = linesRDD.toJavaRDD().map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String row) throws Exception {
                String line = row.replaceAll("\\?", "999999.0");
                String[] tokens = line.split(",");
                Integer last = Integer.parseInt(tokens[13]);
                double[] features = new double[13];
                for (int i = 0; i < 13; i++) {
                    features[i] = Double.parseDouble(tokens[i]);
                }
                Vector v = new DenseVector(features);
                Double value = 0.0;
                if (last.intValue() > 0)
                    value = 1.0;
                LabeledPoint lp = new LabeledPoint(value, v);
                return lp;
            }
        });

        double[] weights = {0.7, 0.3};
        long split_seed = 12345L;
        JavaRDD<LabeledPoint>[] split = data.randomSplit(weights, split_seed);
        JavaRDD<LabeledPoint> training = split[0];
        JavaRDD<LabeledPoint> test = split[1];

        ///////////////////// Ranodom forest 91% accuracy /////////////////////
        Integer numClasses = 26;
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        Integer numTrees = 5; // Use more in practice.
        String featureSubsetStrategy = "auto"; // Let the algorithm choose.
        String impurity = "gini";
        Integer maxDepth = 20;
        Integer maxBins = 40;
        Integer seed = 12345;
        final RandomForestModel model = RandomForest.trainClassifier(training, numClasses,categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);
        long model_building_end = System.currentTimeMillis();
        System.out.println("Model building time: " + (model_building_end - model_building_start)+" ms");

        //Save the model for future use
        long model_saving_start = System.currentTimeMillis();
        //String model_storage_loc = "models/heartdiseasesRandomForestModel";
        //model.save(spark.sparkContext(), model_storage_loc);
        long model_saving_end = System.currentTimeMillis();
        System.out.println("Model saving time: " + (model_saving_end - model_saving_start)+" ms");
        //
        //final RandomForestModel model2 = RandomForestModel.load(spark.sparkContext(), model_storage_loc);

        //Calculate the prediction on test set
        JavaPairRDD<Double, Double> predictionAndLabel =
                test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
                    /**
                     *
                     */
                    private static final long serialVersionUID = 1L;

                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        return new Tuple2<>(model.predict(p.features()), p.label());
                    }
                });

        List<Tuple2<Double, Double>> result = predictionAndLabel.collect();
        for(java.util.Iterator<Tuple2<Double, Double>> it = result.iterator(); it.hasNext();){
            System.out.println(it.next());
        }

        //Calculate the accuracy of the prediction
        double accuracy = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
            @Override
            public Boolean call(Tuple2<Double, Double> pl) {
                return pl._1().equals(pl._2());
            }
        }).count() / (double) data.count();

        //Print the prediction accuracy
        System.out.println("Accuracy of the classification: "+accuracy);
        //my_data.show(false);
        spark.stop();
    }
}
