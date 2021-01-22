package org.example;

import org.apache.spark.sql.SparkSession;

public class UtilityForSparkSession {
	public static SparkSession mySession() {
		SparkSession spark = SparkSession		
	            	.builder()
	      		.master("local[*]")
	      		.config("spark.sql.warehouse.dir", "E:/Exp/") // change accordingly
	      		.appName("Bitcoin Preprocessing")
	      		.getOrCreate();
		return spark;
	}

}
