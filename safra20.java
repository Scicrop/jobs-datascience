//Criado por Luiz Carlos
// https://github.com/zolpy/jobs-datascience
//
package telas.Safra2020;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static javafx.application.Application.launch;

public class safra20 {
    private static Instances instancias;
    //---------------------------------------------------------------------------------------------------------------
    static File Estimativa_de_Insetos = new File("C:\\Users\\Luiz Carlos\\Downloads\\sciCrop\\jobs-datascience-master\\Nova pasta\\Estimativa_de_Insetos.txt");
    static File Tipo_de_Cultivo = new File("C:\\Users\\Luiz Carlos\\Downloads\\sciCrop\\jobs-datascience-master\\Nova pasta\\Tipo_de_Cultivo.txt");
    static File Tipo_de_Solo = new File("C:\\Users\\Luiz Carlos\\Downloads\\sciCrop\\jobs-datascience-master\\Nova pasta\\Tipo_de_Solo.txt");
    static File Categoria_Pesticida = new File("C:\\Users\\Luiz Carlos\\Downloads\\sciCrop\\jobs-datascience-master\\Nova pasta\\Categoria_Pesticida.txt");
    static File Doses_Semana = new File("C:\\Users\\Luiz Carlos\\Downloads\\sciCrop\\jobs-datascience-master\\Nova pasta\\Doses_Semana.txt");
    static File Semanas_Utilizando = new File("C:\\Users\\Luiz Carlos\\Downloads\\sciCrop\\jobs-datascience-master\\Nova pasta\\Semanas_Utilizando.txt");
    static File Semanas_Sem_Uso = new File("C:\\Users\\Luiz Carlos\\Downloads\\sciCrop\\jobs-datascience-master\\Nova pasta\\Semanas_Sem_Uso.txt");
    static File Temporada = new File("C:\\Users\\Luiz Carlos\\Downloads\\sciCrop\\jobs-datascience-master\\Nova pasta\\Temporada.txt");
    //---------------------------------------------------------------------------------------------------------------
    public static void main(String[] args) throws Exception {

        File arquivo = new File("C:\\Users\\Luiz Carlos\\Downloads\\sciCrop\\jobs-datascience-master\\weka_file_safra_2020.arff");
        FileOutputStream f = new FileOutputStream(arquivo);
        int nd = 8858;
        Integer[] do_Estimativa_de_Insetos = new Integer[nd];
        Integer[] do_Tipo_de_Cultivos = new Integer[nd];
        Integer[] do_Tipo_de_Solo = new Integer[nd];
        Integer[] do_Categoria_Pesticida = new Integer[nd];
        Integer[] do_Doses_Semana = new Integer[nd];
        Integer[] do_Semanas_Utilizando = new Integer[nd];
        Integer[] do_Semanas_Sem_Uso = new Integer[nd];
        Integer[] do_Temporada = new Integer[nd];
        String exportacao1 = "@relation Safra_2020\n\n";
        String exportacao2 = "@attribute Estimativa_de_Insetos real\n" +
                "@attribute Tipo_de_Cultivo {0,1}\n" +
                "@attribute Tipo_de_Solo {0,1}\n" +
                "@attribute Categoria_Pesticida {1,2,3}\n" +
                "@attribute Doses_Semana real\n" +
                "@attribute Semanas_Utilizando real\n" +
                "@attribute Semanas_Sem_Uso real\n" +
                "@attribute Temporada {1,2,3}\n" +
                "@attribute dano_na_plantacao {0,1,2} \n\n";
        String exportacao4 = "@data\n";
        String exportacao5 = "";


        BufferedReader br = null;
        try {

            br = new BufferedReader(new FileReader(Estimativa_de_Insetos));
            String str;
            int z = 0;
            while ((str = br.readLine()) != null) {
                do_Estimativa_de_Insetos[z] = Integer.parseInt(str);
//                System.out.println("vetor ["+ z +"]: "+ Integer.parseInt(str) + " vetor ["+ z +"]:" + do_Estimativa_de_Insetos [z] );
                z = z + 1;
            }

            z = 0;
            br = new BufferedReader(new FileReader(Tipo_de_Cultivo));
            while ((str = br.readLine()) != null) {
                do_Tipo_de_Cultivos[z] = Integer.parseInt(str);
//                System.out.println("vetor ["+ z +"]: "+ Integer.parseInt(str) + " vetor ["+ z +"]:" + do_Estimativa_de_Insetos [z] );
                z = z + 1;
            }

            z = 0;
            br = new BufferedReader(new FileReader(Tipo_de_Solo));
            while ((str = br.readLine()) != null) {
                do_Tipo_de_Solo[z] = Integer.parseInt(str);
//                System.out.println("vetor ["+ z +"]: "+ Integer.parseInt(str) + " vetor ["+ z +"]:" + do_Estimativa_de_Insetos [z] );
                z = z + 1;
            }

            z = 0;
            br = new BufferedReader(new FileReader(Categoria_Pesticida));
            while ((str = br.readLine()) != null) {
                do_Categoria_Pesticida[z] = Integer.parseInt(str);
//                System.out.println("vetor ["+ z +"]: "+ Integer.parseInt(str) + " vetor ["+ z +"]:" + do_Estimativa_de_Insetos [z] );
                z = z + 1;
            }

            z = 0;
            br = new BufferedReader(new FileReader(Doses_Semana));
            while ((str = br.readLine()) != null) {
                do_Doses_Semana[z] = Integer.parseInt(str);
//                System.out.println("vetor ["+ z +"]: "+ Integer.parseInt(str) + " vetor ["+ z +"]:" + do_Estimativa_de_Insetos [z] );
                z = z + 1;
            }

//            FileReader fr5 = new FileReader(Semanas_Utilizando);
            br = new BufferedReader(new FileReader(Semanas_Utilizando));
            z = 0;
            while ((str = br.readLine()) != null) {
                do_Semanas_Utilizando[z] = Integer.parseInt(str);
//                System.out.println("vetor ["+ z +"]: "+ Integer.parseInt(str) + " vetor ["+ z +"]:" + do_Estimativa_de_Insetos [z] );
                z = z + 1; //z+=1;
            }

//            FileReader fr6 = new FileReader(Semanas_Sem_Uso);
            br = new BufferedReader(new FileReader(Semanas_Sem_Uso));
            z = 0;
            while ((str = br.readLine()) != null) {
                do_Semanas_Sem_Uso[z] = Integer.parseInt(str);
//                System.out.println("vetor ["+ z +"]: "+ Integer.parseInt(str) + " vetor ["+ z +"]:" + do_Estimativa_de_Insetos [z] );
                z = z + 1;
            }

//            FileReader fr7 = new FileReader(Temporada);
            br = new BufferedReader(new FileReader(Temporada));
            z = 0;
            while ((str = br.readLine()) != null) {
                do_Temporada[z] = Integer.parseInt(str);
//                System.out.println("vetor ["+ z +"]: "+ Integer.parseInt(str) + " vetor ["+ z +"]:" + do_Estimativa_de_Insetos [z] );
                z = z + 1;
            }


        } catch (IOException e) {
            System.out.println("Arquivo não encontrado!");
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }


        try {
        ObjectInputStream modelo = new ObjectInputStream(new FileInputStream("C:\\Users\\Luiz Carlos\\Downloads\\sciCrop\\jobs-datascience-master\\model\\IBk.model"));
        IBk Mod = (IBk) modelo.readObject();
        modelo.close();
        System.out.println("------------------------------------------acabou_modeloRF");

        carregaBaseWeka();
        System.out.println("------------------------------------------acabou_carregaBaseWeka()");
        //Novo Registro
        Instance novo = new DenseInstance(instancias.numAttributes());
        novo.setDataset(instancias);
        f.write(exportacao1.getBytes());
        f.write(exportacao2.getBytes());
        f.write(exportacao4.getBytes());
        System.out.println("------------------------------------------acabou_f.write(exportacao4.getBytes());");


        for (int i = 0; i < nd; i++) {
            novo.setValue(0, Math.round(do_Estimativa_de_Insetos[i]));
            novo.setValue(1, Math.round(do_Tipo_de_Cultivos[i]));
            novo.setValue(2, Math.round(do_Tipo_de_Solo[i]));
            novo.setValue(3, Math.round(do_Categoria_Pesticida[i]));
            novo.setValue(4, Math.round(do_Doses_Semana[i]));
            novo.setValue(5, Math.round(do_Semanas_Utilizando[i]));
            novo.setValue(6, Math.round(do_Semanas_Sem_Uso[i]));
            novo.setValue(7, Math.round(do_Temporada[i]));
//            System.out.println("do_Estimativa_de_Insetos[i]: " + do_Estimativa_de_Insetos[i]);

            double[] resultado = Mod.distributionForInstance(novo);
            exportacao5 = (Math.round(novo.value(0)) + "," +
                           Math.round(novo.value(1)) + "," +
                           Math.round(novo.value(2)) + "," +
                           Math.round(novo.value(3)) + "," +
                           Math.round(novo.value(4)) + "," +
                           Math.round(novo.value(5)) + "," +
                           Math.round(novo.value(6)) + "," +
                           Math.round(novo.value(7)) + "," +
                           Math.round(resultado[0]) + "\n");
            f.write(exportacao5.getBytes());
        }

        System.out.println("------------------------------------------acabou_f.write(exportacao5.getBytes());");


        f.close();


        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static void carregaBaseWeka() throws Exception {
        ConverterUtils.DataSource fonte = new ConverterUtils.DataSource("C:\\Users\\Luiz Carlos\\Downloads\\sciCrop\\jobs-datascience-master\\Esqueleto_Safra_2020.arff"); //o arff tem 50 atributos e 1 classe mas
        instancias = fonte.getDataSet();
        instancias.setClassIndex(instancias.numAttributes() - 1); //setando a ultima coluna
    }


}
