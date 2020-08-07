//Criado por Luiz Carlos
// https://github.com/zolpy/jobs-datascience
//
package telas.Safra2020;


import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class EscolheModelo {

    //------------------------------------------------------------------------------------------------------------.arff
    public static void main(String args[]) throws Exception {
        DataSource fonte = new DataSource("C:\\Users\\Luiz Carlos\\Downloads\\sciCrop\\jobs-datascience-master\\Safra_2018_2019.arff");

        int folds = 10;
        int vezes = 5;
        Instances dados = fonte.getDataSet();
        dados.setClassIndex(dados.numAttributes() - 1);

//-----------------------------------------------------------------------------
        Classifier classificadorMLP = new MultilayerPerceptron();
        System.out.println("--------------------------------------------began_MLP");
        for (int seed = 1; seed <= vezes; seed++) {
            Evaluation avaliador = new Evaluation(dados);
            avaliador.crossValidateModel(classificadorMLP, dados, folds, new Random(seed));

            System.out.println(String.valueOf(avaliador.pctCorrect()).replace('.', ','));
        }
        System.out.println("--------------------------------------------end_MLP");
//-----------------------------------------------------------------------------
        Classifier classificadorJ48 = new J48();
        System.out.println("--------------------------------------------Began_J48");
        for (int seed = 1; seed <= vezes; seed++) {
            Evaluation avaliador = new Evaluation(dados);
            avaliador.crossValidateModel(classificadorJ48, dados, folds, new Random(seed));

            System.out.println(String.valueOf(avaliador.pctCorrect()).replace('.', ','));
        }
        System.out.println("--------------------------------------------end_J48");

//-----------------------------------------------------------------------------
        Classifier classificadorNaiveBayes = new NaiveBayes();
        System.out.println("--------------------------------------------Began_NaiveBayes");
        for (int seed = 1; seed <= vezes; seed++) {
            Evaluation avaliador = new Evaluation(dados);
            avaliador.crossValidateModel(classificadorNaiveBayes, dados, folds, new Random(seed));

            System.out.println(String.valueOf(avaliador.pctCorrect()).replace('.', ','));
        }
        System.out.println("--------------------------------------------end_NaiveBayes");

//-----------------------------------------------------------------------------
        Classifier classificadorIBk = new IBk(1);
        System.out.println("--------------------------------------------Began_IBk");
        for (int seed = 1; seed <= vezes; seed++) {
            Evaluation avaliador = new Evaluation(dados);
            avaliador.crossValidateModel(classificadorIBk, dados, folds, new Random(seed));

            System.out.println(String.valueOf(avaliador.pctCorrect()).replace('.', ','));
        }
        System.out.println("--------------------------------------------end_IBk");
//-----------------------------------------------------------------------------
        Classifier classificadorRandomForest = new RandomForest();
        System.out.println("--------------------------------------------Began_RandomForest");
        for (int seed = 1; seed <= vezes; seed++) {
            Evaluation avaliador = new Evaluation(dados);
            avaliador.crossValidateModel(classificadorRandomForest, dados, folds, new Random(seed));

            System.out.println(String.valueOf(avaliador.pctCorrect()).replace('.', ','));
        }
        System.out.println("--------------------------------------------end_RandomForest");
//-----------------------------------------------------------------------------
        Classifier classificadorRandomTree = new RandomTree();
        System.out.println("--------------------------------------------Began_RandomTree");
        for (int seed = 1; seed <= vezes; seed++) {
            Evaluation avaliador = new Evaluation(dados);
            avaliador.crossValidateModel(classificadorRandomTree, dados, folds, new Random(seed));

            System.out.println(String.valueOf(avaliador.pctCorrect()).replace('.', ','));
        }
        System.out.println("--------------------------------------------end_RandomTree");
//-----------------------------------------------------------------------------


    }
}
