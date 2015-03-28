package regression;
//package files;

import files.FileManager;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import matrix.Matrix;
import matrix.MatrixMathematics;
import matrix.NoSquareException;

/**
 * 
 * @author ata
 *
 * This class provide methods to calculate the coefficients in a multi-linear regression modelling.
 *
 */
public class MultiLinear {

	private Matrix X;
	final private Matrix Y;
	final private boolean bias;
        double [][] dataX;
        double [] dataY;
	
	public MultiLinear(final Matrix x, final Matrix y) {
		this(x,y,true);
	}
	
	public MultiLinear(final Matrix x, final Matrix y, final boolean bias) {
		super();
		this.X = x;
		this.Y = y;
		this.bias = bias;
	}
        
        public static void main(String [] args) throws NoSquareException{
//            FileManager fm = new FileManager();
//            
//            Matrix X = new Matrix(dataX);
//            Matrix Y = new Matrix(dataY);
//            MultiLinear ml = new MultiLinear(X, Y);
//            Matrix beta = ml.calculate();
            
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
        
        
	/**
	 * beta a matrix which gives the coefficients of a multi-linear regression equation Y = sum_of(beta_i * x_i) 
	 * beta = Inverse_of(X_transpose * X) * X_transpose * Y
	 * @return
	 * @throws NoSquareException
	 */
	public Matrix calculate() throws NoSquareException {
		if (bias)
			this.X = X.insertColumnWithValue1();
		checkDiemnsion();
		final Matrix Xtr = MatrixMathematics.transpose(X); //X'
		final Matrix XXtr = MatrixMathematics.multiply(Xtr,X); //XX'
		final Matrix inverse_of_XXtr = MatrixMathematics.inverse(XXtr); //(XX')^-1
		if (inverse_of_XXtr == null) {
			System.out.println("Matrix X'X does not have any inverse. So MLR failed to create the model for these data.");
			return null;
		}
		final Matrix XtrY = MatrixMathematics.multiply(Xtr,Y); //X'Y
		return MatrixMathematics.multiply(inverse_of_XXtr,XtrY); //(XX')^-1 X'Y
	}

	/**
	 * Preconditions checks
	 */
	void checkDiemnsion() {
		if (X == null) 
			throw new NullPointerException("X matrix cannot be null.");
		if (Y == null) 
			throw new NullPointerException("Y matrix cannot be null.");
		
		if (X.getNcols() > X.getNrows()) {
			throw new IllegalArgumentException("The number of columns in X matrix (descriptors) cannot be more than the number of rows");
		}
		if (X.getNrows() != Y.getNrows()) {
			throw new IllegalArgumentException("The number of rows in X matrix should be the same as the number of rows in Y matrix. ");
		}
	}
	
}
