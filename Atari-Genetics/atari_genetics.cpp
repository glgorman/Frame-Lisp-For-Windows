
#include "stdafx.h"
#include <vector>
#include <math.h>
#include <time.h>

#include "../Frame Lisp/intrinsics.h"
#include "../Frame Lisp/text_object.h"
#include "atari_genetics.h"

int atari::main()
{
	int t1,t2;
	int iter = ITERATIONS;
	int m,n;
	bool verbose = true;

	// first initialize the dataset
	vector<neural::dataset> datasets;
	datasets.resize (NUMBER_OF_MODELS);    
	for (n=0;n<(int)datasets.size();n++)
	{
		datasets[n].bind (functions::square);
		datasets[n].initialize (NP);
	}
	// now construct the neural network
    vector<neural::gnn> models;
	models.resize (NUMBER_OF_MODELS);
	for (n=0;n<(int)models.size();n++)
	{
		models[n].bind (&datasets[n]);
		models[n].init (n,iter);
	}
	t1 = GetTickCount();

	bool result;
	models[0].report_eval (false);
	for (n=0;n<iter;n++)
	for (m=0;m<(int)models.size();m++)
	{
		result = models[m].train (n);
		if (result==true)
		{
			models[m].store_solution(n);
		}
		if ((result==true)&&(verbose==true))
		{
			writeln (output,"training data stored ... ");
			models[m].report(n);
		}
	}
	t2 = GetTickCount();
	writeln(output);
	writeln (output,"Training took ",(t2-t1)," msec.");
	neural::report_weights (models[0].w1);
	models[0].report_eval (true);
    
	return 0;
}

void neural::gnn::store_solution (int N)
{
	dist = sqrt(dist/NP);
	DB = dist;
	F -= DF;
	neural::copy_weights (w1,w0);
}

void neural::gnn::report (int iter)
{
	bool checkpoint = false;
	writeln (output);

	if (checkpoint==true)
		writeln (output,"Checkpoint: ",iter);
	else
		writeln (output, "Model: ",m_index," Iteration: ", iter, " ",dist, " ", DB);

	report_weights (w1);
	report_eval (true);
}

void neural::gnn::report_eval (bool flag)
{
	double arg,value;
	double x,y;
	int i;
	if (flag==false)
//		writeln (output,"Model: ",m_index," Initial Values:");
		writeln (output,"Initial Values:");
	else
		writeln (output,"Model: ",m_index," Current Values:");

	for (i=0; i<NP; i++)
	{
		x = m_ds->x(i);
		y = m_ds->y(i);
		arg = double(i)/m_ds->size();
		value = eval0(w1,i);
		writeln (output," >>> X=",x," Y=",y, " G=",value);
    }
};

void neural::report_weights (const vector<double> &data)
{
	int i;
	size_t sz = data.size ();
	for (i=0; i<(int)sz; i++)
	{
		writeln (output," >>> w1[",i,"] = ",data[i]);
	}
};

inline void neural::copy_weights (vector<double> &dst, const vector<double> &src)
{
	size_t sz0 = src.size ();
	size_t sz1 = dst.size ();
	ASSERT (sz0==sz1);
	for (int i=0; i<(int)sz0; i++)
		dst[i] = src[i];
}

void neural::tensor::resize (int rows, int cols)
{
	m_rows = rows;
	m_cols = cols;
	m_weight.resize(rows*cols);
}

double &neural::tensor::w (int r, int c)
{
	return *(&m_weight[r*m_cols+c]);
}

void neural::dataset::initialize (int sz)
{
	// Initialize points
	int i;
	m_size = sz;
	x0.resize (sz);
	y0.resize (sz);
	DB = 1.0/sz;
	for (i=0; i<sz; i++)
	{
        x0[i] = i* DB;
        y0[i] = m_fptr (x0[i]);
	}
}

bool neural::dataset::bind (double(*arg)(double))
{
	m_fptr=arg;
	if (m_fptr!=NULL)
		return true;
	else
		return false;
}

double functions::sigmoid (double arg)
{
	double result;
	result = 1.0/(1+exp(-arg));
	return result;
}

double functions::square (double arg)
{
	double result;
	result = arg*arg;
	return result;
}

neural::gnn::gnn ()
{  
	m_best = -1;
	m_gain1 = 2.0;
	m_gain2 = 2.0;
	m_gain3 = 5.0;
	m_gain4 = 3.0;

	size_t NH = 7;
	size_t NM = 2*NH;
	
	DB_INIT = 10000.0;
	F_INIT = 2.5;
	w0.resize(NM);
    w1.resize(NM);
	m_out0.resize(NM);
	m_fxfr = &functions::sigmoid;
}

void neural::gnn::bind (dataset *ds)
{
	m_ds = ds;
}

double neural::gnn::distance ()
{
	return dist;
}

double neural::gnn::db ()
{
	return DB;
}

void neural::gnn::init (int index,int iter)
{
	int i;
	m_index = index;
	
	F = F_INIT;
	DF = F/iter;
	DB = DB_INIT;
	size_t NM = w1.size ();

	// Initialize w1 with random values
	srand((unsigned int)time(NULL));
    for (i = 0; i<(int)NM; i++)
        w1[i] = (double)rand() / RAND_MAX;
}

void neural::gnn::randomize ()
{
	int i;
	size_t NM = w1.size ();
	for (i=0;i<(int)NM;i++)
        w0[i] = w1[i] + m_gain3*F*((double)rand()/RAND_MAX-0.5);
}

bool neural::gnn::train (int N)
{
    int k;
	double val, delta;
    dist = 0.0;
	randomize();
	size_t sz = m_ds->size();
    for (k=0;k<(int)sz;k++)
	{
		val = eval0(w0,k);
		delta = val-m_ds->y(k);
		dist += delta*delta;
    }
	if (DB>dist)
		return true;
	else
		return false;
}

double neural::gnn::get_input (int k)
{
	double result;
	result = m_ds->x(k);
	return result;
}

double neural::gnn::eval0 (const vector<double> &w, int k)
{
	int i,j;
	size_t NN = m_out0.size();
	size_t NH = w1.size ();
	size_t NM = NH>>1;
	double sum,temp0;
	double m_bias = 0.5;

	// first layer??
//	writeln (output,"eval0: k=",k);
	for (j=0;j<(int)NN; j++)
	{
		sum = 0.0;
		for (i=0;i<(int)NH;i++)
		{
			temp0 = w[i]*get_input(k);
//			writeln (output," >>> w[",i,"]=",w[i],"*",get_input(k),"=",temp0);
//			writeln (output," >>> sum+=",m_fxfr(temp0));
			sum+= m_fxfr(temp0)-m_bias;
		}
		m_out0[j] = m_fxfr(sum*m_gain1);
//		writeln (output," >>> SUM =",sum,", OUTPUT=",m_out0[j]);
	}
	
	// second layer??
	sum = 0.0;
	for (i=0; i<(int)NM; i++)
	{
		sum+=w[i+NM]*m_out0[i];
	}
	m_out = m_gain4*m_fxfr(sum*m_gain2);
	return m_out;
}

double neural::model::eval1 (double val)
{
	int i,j;
	double sum;
	size_t NN = out.size();
	size_t NH = w1.size ();
	size_t NM = NH>>1;
	double &result = out[NN-1];

	// first layer??
	result = 0.0;
  	for (j=0; j<(int)(NN-1); j++)
	{
		sum = 0.0;
		for (i=0;i<(int)NH;i++)
			sum+=w1[i]*val;

		out[j] = m_fxfr(sum*m_gain1);
	}
	// second layer??
	for (i=0; i<(int)NM; i++)
	{
		out[NN-1]+=w1[i+NM]*out[i];
	}
	out[NN-1] = m_fxfr(out[NN-1]*m_gain2);
	result = out[NN-1];
	return result;
}

#if 0
// Adjust hyperparameters
#define NP 24
#define NUMBER_OF_MODELS 32
#define ITERATIONS 16384
#define MUTATION_RATE 0.1
#define CROSSOVER_RATE 0.7

// Change activation function to ReLU
double relu(double arg) {
    return std::max(0.0, arg);
}

// Initialize weights with Xavier initialization
void neural::gnn::init(int index, int iter) {
    int i;
    m_index = index;
    F = F_INIT;
    DF = F / iter;
    DB = DB_INIT;
    size_t NM = w1.size();
    // Initialize w1 with random values
    srand(time(NULL));
    for (i = 0; i < (int)NM; i++) {
        w1[i] = (double)rand() / RAND_MAX * sqrt(2.0 / NM);
    }
}

// Use MSE as the fitness function
double neural::gnn::distance() {
    double dist = 0.0;
    size_t sz = m_ds->size();
    for (int k = 0; k < (int)sz; k++) {
        double val = eval0(w0, k);
        double delta = val - m_ds->y(k);
        dist += delta * delta;
    }
    return dist;
}

// Use tournament selection
bool neural::gnn::train(int N) {
    int k;
    double val, delta;
    dist = 0.0;
    randomize();
    size_t sz = m_ds->size();
    for (k = 0; k < (int)sz; k++) {
        val = eval0(w0, k);
        delta = val - m_ds->y(k);
        dist += delta * delta;
    }
    if (DB > dist) {
        return true;
    } else {
        return false;
    }
}

#endif
