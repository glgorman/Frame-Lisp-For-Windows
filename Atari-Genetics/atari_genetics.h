


#ifndef PI
#define PI (3.1415926535897932384)
#endif

#define NP 24
#define NUMBER_OF_MODELS  32
#define ITERATIONS	16384
#define DATA_TYPE	vector<double>

class text_object;

namespace atari
{
	void train_ann();
	int main();
};

namespace neural
{
class gnn;
class dataset;
class model;
class layer;
class tensor;

	void copy_weights (DATA_TYPE &dst, const DATA_TYPE &src);
	void report_weights (const vector<double> &data);
}

namespace functions
{
	double square (double);
	double sigmoid (double);
}

class neural::dataset
{
protected:
	int m_size;
	double DB;
	double (*m_fptr)(double);	
	DATA_TYPE x0, y0;

public:
	void initialize (int);
	bool bind (double(*fptr)(double));
	inline size_t size () { return m_size; }
	inline double x(int k) { return x0[k];}
	inline double y(int k) { return y0[k];}
};

class neural::tensor
{
protected:
	size_t m_rows,m_cols;
	DATA_TYPE m_weight;

public:
	void resize (int rows, int cols);
	inline double &w (int r, int c);
};

class neural::layer
{
friend class neural::model;
protected:	
	int	m_size;
	double (*m_fxfr)(double);	
    DATA_TYPE m_weight;
	DATA_TYPE m_bias;
	DATA_TYPE m_input;
	DATA_TYPE m_output;

public:
	bool bind (double(*fptr)(double));
	void initialize (int);
	void propagate ();
};

class neural::model
{
friend class neural::gnn;
protected:
	size_t sz;
	double m_gain1,m_gain2;
	double m_gain3,m_gain4;	
	double (*m_fxfr)(double);	
    DATA_TYPE w0, w1, out;

public:
	void initialize (int);
	void clone (model &);
	double eval1 (double val);
	size_t size () { return sz; }
};

class neural::gnn
{
friend class atari;
protected:
	int m_best;
	int m_index;
	dataset *m_ds;
	double DB_INIT, F_INIT;
	double F, DB, DF,dist;
	double m_gain1,m_gain2;
	double m_gain3,m_gain4;	
	double (*m_fxfr)(double);	
	double m_out;

public:
	DATA_TYPE w0, w1, m_out0;
	gnn ();
	void init (int n, int i);
	void bind (dataset *ds);
	void report (int iter);
	void report_eval (bool);
	inline bool train(int iter);
	inline double distance ();
	inline double db ();
	void store_solution (int N);

	void load (neural::model &);
	void save (neural::model &);
	
protected:
	void randomize ();
	double get_input (int k);
	double eval0 (const vector<double> &w, int k);
	double eval1 (double);	
	void report2 ();
};


