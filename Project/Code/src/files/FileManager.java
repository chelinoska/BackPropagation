package files;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Celina
 */
public class FileManager {
    
    double [][] dataX;
    double [] dataY;
    
    //Load the training set methods
    public FileManager(){
        
    }
    
    public void readFile(String fileName) throws FileNotFoundException, IOException{
        
        
        BufferedReader br = new BufferedReader(new FileReader("turbine.txt"));
            
        try {
            StringBuilder sb = new StringBuilder();
            String line = br.readLine();
            int nx = line.split(line).length-1;
            dataX = new double[450][nx];
            dataY = new double[450];
            int i = 0;
            
            while (line != null) {
                
                sb.append(line);
                sb.append("\n");
                line = br.readLine();
                String [] values = line.split(" ");
                
                for (int j = 0; j < values.length; j++) {
                    String string = values[i];
                    if(j!=values.length-1)
                        dataX[i][j] = Double.parseDouble(string);
                    else
                        dataY[i] = Double.parseDouble(string);
                }
                i++;
            }
            String everything = sb.toString();
        }catch(Exception e){
            e.toString();
        }finally {
            br.close();
        }
    }
    
   
   
    
    // Load the network structure
    // Define a standard input file
    
    
}
