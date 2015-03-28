
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * @author Celina
 */

public class NeuralNetwork {
    
    int inputPat;
    int inputPred;
    int inputVar;
    int toPredict;
    int repetitions;
    int epochs;
    int numLayers;
    float learningRate;
    float momentum;
    int [] neuronsPerLayer;
    float [][] trainInPatterns;
    float [][] trainInScaledPatterns;
    float [][] trainOutPatterns;
    float [][] trainOutScaledPatterns;
    float [][] inputPatterns; // tamI = neurons -- tamJ = patterns
    float [][] outputPatterns;
    float [][] inputScaledPatterns; // tamI = neurons -- tamJ = patterns
    float [][] outputScaledPatterns;
    ArrayList <float[][]> weights;
    ArrayList <float[]> thresholds;
    float inputMax[];
    float inputMin[];
    float outputMax[];
    float outputMin[];
        
    public NeuralNetwork(){
    }
    
    public NeuralNetwork(String fileName){
            this.readFile(fileName);
    }
    
    public void readFile(String fileName){
        
        fileName = getClass().getResource(fileName).getFile().replace("%20", " ");
        System.out.println("FileName  : "+fileName);
        
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(fileName));
        } catch (FileNotFoundException ex) {
            Logger.getLogger(NeuralNetwork.class.getName()).log(Level.SEVERE, null, ex);
        }
            
        try {
            StringBuilder sb = new StringBuilder();
            inputPat = Integer.valueOf(br.readLine().split(" ")[1]);
            inputPred = Integer.valueOf(br.readLine().split(" ")[1]);
            inputVar = Integer.valueOf(br.readLine().split(" ")[1]);
            toPredict = Integer.valueOf(br.readLine().split(" ")[1]);
            numLayers = Integer.valueOf(br.readLine().split(" ")[1]);
            inputPatterns = new float[inputPat][inputVar];
            outputPatterns = new float[inputPat][toPredict];
            inputScaledPatterns = new float[inputPat][inputVar];
            outputScaledPatterns = new float[inputPat][toPredict];
            trainInPatterns = new float[inputPred][inputVar];
            trainInScaledPatterns = new float[inputPred][inputVar];
            trainOutPatterns = new float[inputPred][toPredict];
            trainOutScaledPatterns = new float[inputPred][toPredict];
            neuronsPerLayer = new int[numLayers];
            
            inputMax = new float[inputVar];
            inputMin = new float[inputVar];
            outputMax = new float[toPredict];
            outputMin = new float[toPredict];
            
            String [] neuronsL = br.readLine().split(" ");
            for (int i = 0; i < neuronsPerLayer.length; i++) {
                neuronsPerLayer[i] = Integer.valueOf(neuronsL[i+1]);
            }
            epochs = Integer.valueOf(br.readLine().split(" ")[1]);
            repetitions = Integer.valueOf(br.readLine().split(" ")[1]);
            learningRate = Float.valueOf(br.readLine().split(" ")[1]);
            momentum = Float.valueOf(br.readLine().split(" ")[1]);
                        
            br.readLine();
            String line = br.readLine();
            
            //Reading patterns
            for(int i=0; i<inputPat; i++){
                String values[] = line.split(" ");
                for (int j = 0; j < inputVar; j++) {
                    inputPatterns[i][j] = Float.valueOf(values[j]);
                }
                int k = inputVar;
                for (int j = 0; j<toPredict; j++){
                    outputPatterns[i][j] = Float.valueOf(values[k]);
                    k++;
                }
                line = br.readLine();
            }
            
            
            for (int i = 0; i < inputPred; i++) {
                line = br.readLine();
            
                String values[] = line.split(" ");
                for (int j = 0; j < inputVar; j++) {
                    trainInPatterns[i][j] = Float.valueOf(values[j]);
                }
                int k = inputVar;
                for (int j = 0; j<toPredict; j++){
                    trainOutPatterns[i][j] = Float.valueOf(values[k]);
                    k++;
                }
            }
            
        }catch(Exception e){
            System.out.println("Exception: "+e.toString());
        }finally {
            try {
                br.close();
            } catch (IOException ex) {
                Logger.getLogger(NeuralNetwork.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        weights = new <float[][]> ArrayList();
        thresholds = new <float[]> ArrayList();
                
        weights.add(new float[inputVar][neuronsPerLayer[0]]);
        for (int i = 0; i < numLayers-1; i++) {
            weights.add(new float[neuronsPerLayer[i]][neuronsPerLayer[i+1]]);
        }
        weights.add(new float[neuronsPerLayer[numLayers-1]][toPredict]);
        
        for (int i = 0; i < numLayers; i++) {
            thresholds.add(new float[neuronsPerLayer[i]]);
        }
        thresholds.add(new float[toPredict]);
        
    }
    
    public void initializeWeights(){
        
        for (int l = 0; l < weights.size(); l++) {
            for (int i = 0; i < weights.get(l).length; i++) {
                for (int j = 0; j < weights.get(l)[i].length; j++) {
                    weights.get(l)[i][j] = (float)Math.random();
                }
            }
        }
    }
    
    public void initializeThresholds(){
    
        for (int l = 0; l < thresholds.size(); l++) {
            for (int i = 0; i < thresholds.get(l).length; i++) {
                thresholds.get(l)[i] = (float)Math.random();
            }
        }
        
    }
    
    public static void main(String [] args){
            NeuralNetwork nn = new NeuralNetwork();
            nn.readFile("turbine.txt");
            nn.scaling();
    }
    
    
    public void scaling(){
        
        
        for (int i = 0; i < inputMax.length; i++) {
            inputMax[i] = inputPatterns[0][i];
            inputMin[i] = inputPatterns[0][i];
        }
        
        for (int i = 0; i < outputMax.length; i++) {
            outputMax[i] = outputPatterns[0][i];
            outputMin[i] = outputPatterns[0][i];
        }
        
        for (int j = 0; j < inputPatterns[0].length; j++) {
            for (int i = 0; i < inputPatterns.length; i++) {
                if(inputPatterns[i][j]>inputMax[j]){
                    inputMax[j]=inputPatterns[i][j];
                }
                if(inputPatterns[i][j]<inputMin[j]){
                    inputMin[j]=inputPatterns[i][j];
                }
            }
        }
        
        for (int j = 0; j < outputPatterns[0].length; j++) {
            for (int i = 0; i < outputPatterns.length; i++) {
                if(outputPatterns[i][j]>outputMax[j]){
                    outputMax[j]=outputPatterns[i][j];
                }
                if(outputPatterns[i][j]<outputMin[j]){
                    outputMin[j]=outputPatterns[i][j];
                }
            }
        }
        
        for (int j = 0; j < inputPatterns[0].length; j++) {
            for (int i = 0; i < inputPatterns.length; i++) {
                inputScaledPatterns[i][j] = (inputPatterns[i][j]-inputMin[j])/(inputMax[j]-inputMin[j]);
            }
        }
        
        for (int j = 0; j < outputPatterns[0].length; j++) {
            for (int i = 0; i < outputPatterns.length; i++) {
                outputScaledPatterns[i][j] = (outputPatterns[i][j]-outputMin[j])/(outputMax[j]-outputMin[j]);
            }
        }
        
        for (int j = 0; j < trainInPatterns[0].length; j++) {
            for (int i = 0; i < trainInPatterns.length; i++) {
               trainInScaledPatterns[i][j] = (trainInPatterns[i][j]-inputMin[j])/(inputMax[j]-inputMin[j]);
            }
        } 
        
        for (int j = 0; j < trainOutPatterns[0].length; j++) {
            for (int i = 0; i < trainOutPatterns.length; i++) {
                trainOutScaledPatterns[i][j] = (trainOutPatterns[i][j]-outputMin[j])/(outputMax[j]-outputMin[j]);
            }
        }
        
    }
    
}
