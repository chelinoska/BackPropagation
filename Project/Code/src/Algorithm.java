
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;


/**
 * @author Celina
 */

public class Algorithm {

    Network nn;
    float[][] hi;
    float[][] hiChi;
    float[][] dHi;
    float[][][] dWeights;
    float[][] dThresholds;
    float[][][] dAWeights;
    float[][] dAThresholds;
    float outputsEpoch[][];
    
    public Algorithm(Network nn){
        this.nn = nn;
    }
    
    public Algorithm(){
    }
    
    public void backPropagation(float[] patternX, float[] patternZ){
        
//        System.out.println("BackPropagation");
        int l = nn.numLayers;
        float suma[];
        
        dHi = new float[nn.numLayers+1][];
        for (int i = 0; i < nn.numLayers; i++) {
            dHi[i] = new float[nn.neuronsPerLayer[i]];
        }
        dHi[nn.numLayers] = new float[nn.toPredict];
        
        for (int j = 0; j < dHi[dHi.length-1].length; j++) {
            dHi[l][j]=derivativeFunction(hiChi[l][j])*(hiChi[l][j]-patternZ[j]);
            System.out.println(String.format("%.2f=(%.2f)'*(%.2f-%.2f)",dHi[l][j],derivativeFunction(hiChi[l][j]),hiChi[l][j],patternZ[j]));
        }
        
        //Other Layers
        for (int k = nn.numLayers; k > 0; k--) {
            //System.out.println("Weights");
            suma = new float[nn.weights[k].length];
            System.out.println("suma.lenght"+suma.length);
            for (int i = 0; i < nn.weights[k].length; i++) {
                for (int j = 0; j < nn.weights[k][i].length; j++) {
//                        System.out.println("(Wij)"+Wij[i][j]+"*(dHi)"+dHi.get(l)[j]+"="+Wij[i][j]*dHi.get(l)[j]);
                        suma[i] += (nn.weights[k][i][j]*dHi[k][j]);
                        System.out.println(String.format("%d %.2f",i, suma[i]));
                }
            }
            for (int j = 0; j < dHi[k-1].length; j++) {
                    float temp = derivativeFunction(hiChi[k-1][j])*suma[j];
                    dHi[k-1][j] = temp;
                    System.out.println(String.format("dHi=(hiChi)'%.2f*(suma)%.2f = %.2f", derivativeFunction(hiChi[k-1][j]),suma[j],dHi[k-1][j]));
            }
//            System.out.println("Layer: "+(l-1));
           
            //l--;
        }
               
    }
    
    public void computeAlgorithm(){
        //Weights initialization
        float[] patternX;
        float[] patternZ;
        nn.scaling();
        
        for (int i = 0; i < nn.repetitions; i++) {
            nn.initializeWeights();
            nn.initializeThresholds();
            dAWeights = nn.weights;
            dAThresholds = nn.thresholds;
            for (int k = 0; k < nn.epochs; k++) {
//                outputsEpoch = nn.outputPatterns;
//               for (int j = 0; j < nn.inputPat; j++) {
               for (int j = 0; j < 1; j++){
//                  //Select Random Pattern
                    int vpat = (int)Math.ceil(Math.random()*nn.inputPat-1);
                    //int vpat = i;
                    
                    patternX = nn.inputScaledPatterns[vpat];
                    patternZ = nn.outputScaledPatterns[vpat];

                    feedForward(patternX);
//                    outputsEpoch[i] = hiChi.get(hiChi.size()-1);
//                    difference(patternZ, hiChi.get(hiChi.size()-1));
                    backPropagation(patternX, patternZ);
                    calculateDWeights();
                    //updateWeights();
                }
//                System.out.println("Error epoch "+j+": "+totalError(nn.outputPatterns, outputsEpoch));
                
////                //calculate error for each epoch
            }
            makePrediction();
            //Make prediction {train, validation}
        }
    }
    
    public void makePrediction(){
        
        float[] patternX;
        float[] patternZ;
        float[] difs = new float[nn.trainInScaledPatterns.length];
        
        System.out.println("Prediction");
        
        for (int i = 0; i < nn.trainInScaledPatterns.length; i++) {
            patternX = nn.trainInScaledPatterns[i];
            patternZ = nn.trainOutScaledPatterns[i];
            
            feedForward(patternX);
//            for (int j = 0; j < hiChi.get(nn.numLayers).length; j++) {
//                nn.trainOutScaledPatterns[i][j] = hiChi.get(nn.numLayers)[j];
//            }
//            System.out.print("In Pattern");
//            for (int k = 0; k < patternX.length; k++) {
//                System.out.print(""+String.format("[%f]", patternX[k]));
//                //System.out.print(" ["+patternX[i]+"] ");
//            }
//            System.out.print("  Out Pattern");
            for (int k = 0; k < patternZ.length; k++) {
                System.out.print(""+String.format("[%.2f,", patternZ[k]));
                //System.out.print(" ["+patternZ[i]+"] ");
            }
//            System.out.print("  Out Predicted");
            for (int k = 0; k < hiChi[nn.numLayers].length; k++) {
                System.out.print(""+String.format("%.2f]", hiChi[nn.numLayers][k]));
                //System.out.print(" ["+patternZ[i]+"] ");
            }
            
            
            System.out.println("");
         }
        
        
         System.out.println("Pesos finales: ");
         int l = 0;
         for (float[][] Wij: nn.weights) {
            System.out.println("Layer: "+l);
            for (int i = 0; i < Wij.length; i++) {
                for (int j = 0; j < Wij[i].length; j++) {
                    System.out.print(String.format("[%.2f]", Wij[i][j]));
                }
                System.out.println(" ");
            }
            l++;
        }
    
    }
    
    public float suma(float array[]){
        float sum = 0.0F;
        
        for (int i = 0; i < array.length; i++) {
            sum+=array[i];
        }
        
        return sum;
    }
    
    public float[] difference(float desired[], float actual[]){
        
        float[] diff = new float[desired.length];
        
        System.out.print(" Difference: ");
        for (int i = 0; i < desired.length; i++) {
            diff[i] = Math.abs(desired[i]-actual[i]);
            System.out.print(""+String.format("[%f]", diff[i]));
        }
        
        return diff;
    
    }
    
    public float[] resAbs(float patA[], float patB[]){
        for (int i = 0; i < patA.length; i++) {
            patB[i] = Math.abs(patA[i]-patB[i]);
        }
        return patB;
    }
    
    public float totalError(float desired[][], float actual[][]){

        return 0.0F;
   }
    
    
    public void updateWeights(){
        
         for (int k = 0; k < dWeights.length; k++) {
             for (int i = 0; i < dWeights[k].length; i++) {
                 for (int j = 0; j < dWeights[k][i].length; j++) {
                     nn.weights[k][i][j]+=dWeights[k][i][j];
                 }
             }
         }
         
        for (int k = 0; k < dThresholds.length; k++) {
             for (int i = 0; i < dThresholds[k].length; i++) {
                nn.thresholds[k][i]+=dThresholds[k][i];
             }
         }
    }
    
    public void calculateDWeights(){
        
        dWeights = new float[nn.weights.length][][];
        dThresholds = new float[nn.weights.length][];
        
        dWeights[0] = new float[nn.inputVar][nn.neuronsPerLayer[0]];
        for (int i = 0; i < nn.numLayers-1; i++) {
            dWeights[i+1] = new float[nn.neuronsPerLayer[i]][nn.neuronsPerLayer[i+1]];
        }
        dWeights[nn.numLayers] = new float[nn.neuronsPerLayer[nn.numLayers-1]][nn.toPredict];
        for (int i = 0; i < nn.numLayers; i++) {
            dThresholds[i] = new float[nn.neuronsPerLayer[i]];
        }
        dThresholds[nn.numLayers] = new float[nn.toPredict];
        
        
        for (int k = 0; k < dWeights.length; k++) {
            for (int i = 0; i < dWeights[k].length; i++) {
                for (int j = 0; j < dWeights[k][i].length; j++) {
                    dWeights[k][i][j]=(k==0)?
                            (-nn.learningRate)*dHi[k][j]*hi[k][j]:
                            (-nn.learningRate)*dHi[k][j]*hiChi[k-1][j];
                    System.out.println(String.format("[%.2f]*[%.2f]", dHi[k][j], hiChi[k][j]));
                    //dWeights[k][i][j]=dWeights[k][i][j]+nn.momentum*dAWeights[k][i][j];
                }
            }
            
            for (int i = 0; i < dThresholds[k].length; i++) {
                dThresholds[k][i]=(nn.learningRate)*dHi[k][i];
                //dThresholds[k][i]=dThresholds[k][i]*nn.momentum*dAThresholds[k][i];
            }   
        }
        
        int l = 0;
        System.out.println("DWeights");
        for (float[][] Wij: dWeights) {
            System.out.println("Layer: "+l);
            for (int i = 0; i < Wij.length; i++) {
                for (int j = 0; j < Wij[i].length; j++) {
                    System.out.print(String.format("[%.2f]", Wij[i][j]));
                }
                System.out.println(" ");
            }
            l++;
        }
        
        
        dAWeights = dWeights;
        dAThresholds = dThresholds;
        
    }
    
    public float activationFunction(float x){
        double value = 1/(1+Math.exp(-x));
        float valF = (float)value;
        return valF;
    }
    
    public float derivativeFunction(float x){
        double value = (Math.exp(-x))/Math.pow((1+Math.exp(-x)), 2);
        float valF = (float)value;
        return valF;
    }
    
    public void feedForward(float [] pattern){
    
        hi = new float[nn.numLayers+1][];
        hiChi = new float[nn.numLayers+1][];
        for (int i = 0; i < nn.numLayers; i++) {
            hi[i]=new float[nn.neuronsPerLayer[i]];
            hiChi[i] = new float[nn.neuronsPerLayer[i]];
        }
        hi[nn.numLayers] = new float[nn.toPredict];
        hiChi[nn.numLayers] = new float[nn.toPredict];
        
        //int l = 0;
        float suma = 0.0F;
//        System.out.println("FeedForward");
        //for (float[][] Wij : nn.weights) {
//        for (int l = 0; l<nn.weights.length; l++){
//            for (int i = 0; i < nn.weights[l][0].length; i++) {
//                    suma = 0.0F;
//                    for (int j = 0; j < nn.weights[l].length; j++) {
//                        suma += (l==0)?
//                                (nn.weights[l][j][i]*pattern[j]):
//                                ((nn.weights[l][j][i]*hiChi[l-1][j]));//-nn.thresholds.get(l)[i]);
//                    }
//                    suma-=nn.thresholds[l][i];
//                    hi[l][i] = suma;
//                    hiChi[l][i] = activationFunction(suma);
//            }
//            //l++;
//        }
        
            float sum[];
            for (int l = 0; l<nn.weights.length; l++){
                sum = new float[nn.weights[l][0].length];
                for (int i = 0; i < nn.weights[l].length; i++) {
                    //suma = 0.0F;
                    for (int j = 0; j < nn.weights[l][i].length; j++) {
                        sum[j] += (l==0)?
                                (nn.weights[l][i][j]*pattern[i]):
                                ((nn.weights[l][i][j]*hiChi[l-1][i]));//-nn.thresholds.get(l)[i]);
                    }
                    for (int j = 0; j < nn.weights[l][i].length; j++) {
                        sum[j]-=nn.thresholds[l][j];
                        hi[l][j] = sum[j];
                        hiChi[l][j] = activationFunction(sum[j]);
                    }
                }
            }
        
//        System.out.println("");
//        System.out.print("In Pattern");
//        for (int i = 0; i < pattern.length; i++) {
//            System.out.print(""+String.format("[%.2f]", pattern[i]));
//            //System.out.print(" ["+patternX[i]+"] ");
//        }
//        System.out.print("  Out Pattern");
//        for (int i = 0; i < patternZ.length; i++) {
//            System.out.print(""+String.format("[%f]", patternZ[i]));
//            //System.out.print(" ["+patternZ[i]+"] ");
//        }
//        System.out.print("  Out Predicted");
//        for (int i = 0; i < hiChi[nn.numLayers].length; i++) {
//            System.out.print(""+String.format("[%.2f]", hiChi[nn.numLayers][i]));
//            //System.out.print(" ["+patternZ[i]+"] ");
//        }
        
        
    }
    
    public static void main(String[] args){
        
        Network nn = new Network("files/turbine.txt");
        Algorithm bp = new Algorithm(nn);
        bp.computeAlgorithm();
     
    }
    
}
