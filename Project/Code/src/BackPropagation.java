
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;


/**
 * @author Celina
 */

public class BackPropagation {

    NeuralNetwork nn;
    ArrayList <float[]> hi = new <float[]>ArrayList();
    ArrayList <float[]> hiChi = new <float[]>ArrayList();
    ArrayList <float[]> dHi = new <float[]>ArrayList();
    ArrayList <float[][]> dWeights = new <float[][]> ArrayList();
    ArrayList <float[]> dThresholds = new <float[]> ArrayList();
    ArrayList <float[][]> dAWeights = new <float[][]> ArrayList();
    ArrayList <float[]> dAThresholds = new <float[]> ArrayList();
    float error[];
    float outputsEpoch[][];
    
    public BackPropagation(NeuralNetwork nn){
        this.nn = nn;
    }
    
    public BackPropagation(){
    }
    
    public void backPropagation(float[] patternX, float[] patternZ){
        
        int l = nn.numLayers;
        float suma[];
        
        dHi = new <float[]>ArrayList();
        for (int i = 0; i < nn.numLayers; i++) {
            dHi.add(new float[nn.neuronsPerLayer[i]]);
        }
        dHi.add(new float[nn.toPredict]);
        
        for (int j = 0; j < dHi.get(dHi.size()-1).length; j++) {
            dHi.get(l)[j]=derivativeFunction(hi.get(l)[j])*(hiChi.get(l)[j]-patternZ[j]);
        }
        
        //Other Layers
        for (int k = nn.weights.size()-1; k > 0; k--) {
            
            suma = new float[dHi.get(l-1).length];
            for (int i = 0; i < suma.length; i++) {
                suma[i] = 0.0f;
            }
            
            for (int i = 0; i < nn.weights.get(l).length; i++) {
                for (int j = 0; j < nn.weights.get(l)[i].length; j++) {
                        suma[i]+=(nn.weights.get(l)[i][j]*dHi.get(l)[j]);
                }
            }
            for (int j = 0; j < dHi.get(l-1).length; j++) {
                dHi.get(l-1)[j] = derivativeFunction(hi.get(l-1)[j])*suma[j];
            }
            l--;
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
            
            float[] desired = new float[nn.inputPat];
            float[] predict = new float[nn.inputPat];
            
            for (int k = 0; k < nn.epochs; k++) {
//            int k = 0;
//            do{
                for (int j = 0; j < nn.inputPat; j++) {
//                  //Select Random Pattern
                    int vpat = (int)Math.ceil(Math.random()*nn.inputPat-1);

                    patternX = nn.inputScaledPatterns[vpat];
                    patternZ = nn.outputScaledPatterns[vpat];

                    feedForward(patternX);
                    backPropagation(patternX, patternZ);
                    calculateDWeights(patternX);
                    updateWeights();
                    desired[j] = patternZ[0];
                    predict[j] = hiChi.get(hiChi.size()-1)[0];
                }
                k++;   
                //System.out.println(totalError(desired, predict));
            }
            //while(totalError(desired, predict)>1.3f);
            //System.out.println("Epoch "+k+": "+totalError(desired, predict));
            System.out.println("Error Training: "+totalError(desired, predict));
            //Make prediction {train, validation}
            makePrediction();
            //nn.epochs += 50;
        }
    }
    
    public void makePrediction(){
        
        float[] patternX;
        float[] desired = new float [nn.trainInScaledPatterns.length];
        float[] predict = new float [nn.trainInScaledPatterns.length];
        
        System.out.println("Prediction for "+nn.epochs+" epochs");
        
        for (int i = 0; i < nn.trainInScaledPatterns.length; i++) {
            
            patternX = nn.trainInScaledPatterns[i];
            desired[i] = nn.trainOutScaledPatterns[i][0];
            feedForward(patternX);
            predict[i] = hiChi.get(hiChi.size()-1)[0];
            for (int k = 0; k < hiChi.get(hiChi.size()-1).length; k++) {
                System.out.print(""+String.format("%f", hiChi.get(hiChi.size()-1)[k]));
            }
            System.out.println("");
         }
        
        System.out.println("Testing Error: "+this.totalError(desired, predict));
    }
        
    public void updateWeights(){
        
         for (int k = 0; k < dWeights.size(); k++) {
             for (int i = 0; i < dWeights.get(k).length; i++) {
                 for (int j = 0; j < dWeights.get(k)[i].length; j++) {
                     nn.weights.get(k)[i][j]=nn.weights.get(k)[i][j]+dWeights.get(k)[i][j];
                 }
             }
         }
         
        for (int k = 0; k < dThresholds.size(); k++) {
             for (int i = 0; i < dThresholds.get(k).length; i++) {
                nn.thresholds.get(k)[i]=nn.thresholds.get(k)[i]+dThresholds.get(k)[i];
             }
         }
    }
    
    public void calculateDWeights(float[] patternX){
        
        dWeights = new ArrayList<float[][]>();
        for (int i = 0; i < nn.weights.size(); i++) {
            dWeights.add(new float[nn.weights.get(i).length][nn.weights.get(i)[0].length]);
        }
        dThresholds = new ArrayList<float[]>();
        for (int i = 0; i < nn.thresholds.size(); i++) {
            dThresholds.add(new float[nn.thresholds.get(i).length]);
        }

        for (int l = 0; l < dWeights.size(); l++) {
            
            for (int i = 0; i < dWeights.get(l).length; i++) {
                for (int j = 0; j < dWeights.get(l)[i].length; j++) {
                    dWeights.get(l)[i][j]=(l==0)?
                            (-nn.learningRate)*dHi.get(l)[j]*patternX[i]:
                            (-nn.learningRate)*dHi.get(l)[j]*hiChi.get(l-1)[i];
//                    To use momentum uncomment the following line
//                    dWeights.get(l)[i][j]=dWeights.get(l)[i][j]+nn.momentum*dAWeights.get(l)[i][j];
                }
            }
            
            for (int i = 0; i < dThresholds.get(l).length; i++) {
                dThresholds.get(l)[i]=(nn.learningRate)*dHi.get(l)[i];
//                To use momentum uncomment the following line
//                dThresholds.get(l)[i]=dThresholds.get(l)[i]*nn.momentum*dAThresholds.get(l)[i];
            }   
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
    
        hi = new <float[]>ArrayList();
        hiChi = new <float[]>ArrayList();
        for (int i = 0; i < nn.numLayers; i++) {
            hi.add(new float[nn.neuronsPerLayer[i]]);
            hiChi.add(new float[nn.neuronsPerLayer[i]]);
        }
        hi.add(new float[nn.toPredict]);
        hiChi.add(new float[nn.toPredict]);

        for (int l = 0; l<nn.weights.size(); l++){
            for (int i = 0; i < nn.weights.get(l)[0].length; i++) {
                    float suma = 0.0f;
                    for (int j = 0; j < nn.weights.get(l).length; j++) {
                        suma += (l==0)?
                                (nn.weights.get(l)[j][i]*pattern[j]):
                                ((nn.weights.get(l)[j][i]*hiChi.get(l-1)[j]));//-nn.thresholds.get(l)[i]);
                    }
                    suma-=nn.thresholds.get(l)[i];
                    hi.get(l)[i] = suma;
                    hiChi.get(l)[i] = activationFunction(suma);
            }
        }
      
    }
    
    public static void main(String[] args){
        
        NeuralNetwork nn = new NeuralNetwork("files/turbine.txt");
        BackPropagation bp = new BackPropagation(nn);
        bp.computeAlgorithm();
     
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
        
        for (int i = 0; i < desired.length; i++) {
            diff[i] = Math.abs(desired[i]-actual[i]);
        }
        
        return diff;
    
    }
    
    public float[] resAbs(float patA[], float patB[]){
        for (int i = 0; i < patA.length; i++) {
            patB[i] = Math.abs(patA[i]-patB[i]);
        }
        return patB;
    }
    
    public float totalError(float desired[], float actual[]){
        float num = suma(difference(desired, actual));
        float den = suma(actual);
        return (float)num/den*100.0f;
   }
    
}
