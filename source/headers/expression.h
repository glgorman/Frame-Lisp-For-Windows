
#ifndef MATH_TYPE
#define MATH_TYPE float
#endif

namespace FRACTIONS
{
//class wordList;
//class text_object;
char *long2Text (long argument);

class fraction
{
private:
	static long GCF (long M, long N);
	static void reduce (fraction &arg);
	static void signadjust (fraction &arg);

public:
	fraction ();
	fraction (int);
	fraction(int, int);
//	fraction (MATH_TYPE);
	bool operator != (int arg);
	long numerator;
	long denominator;

	fraction operator * (int arg);
	fraction operator * (MATH_TYPE arg);
	fraction operator + (fraction arg);
	fraction operator - (fraction arg);
	fraction operator * (fraction arg);
	fraction operator % (fraction arg);
	fraction operator / (fraction arg);
	fraction operator / (int arg);
	fraction &operator += (fraction arg);
	fraction &operator -= (fraction arg);
	fraction &operator - (void);

	bool operator<(fraction arg);
	bool operator>(fraction arg);
	bool operator<(int arg);
	bool operator>(int arg); 

	float convert_to_real ();
	fraction &operator = (fraction &arg);
	long operator = (long arg);
};

class rational
{
public:
	int mantissa, numerator, denominator;
	rational (fraction);
};
}

namespace mathop
{
	typedef enum opcode
	{		
		null,identifier,integer,real,frac,
		add,subtract,multiply,divide,and,or,
		nand,nor,xor,not,modulus,symbol,
		left,right,function,
	} _opcode;
}

class operation
{
private:
	mathop::opcode m_opcode;

public:
	inline operation &operator = (mathop::opcode x)
	{
		m_opcode = x;
		return *this;
	};
	inline bool operator == (mathop::opcode x)
	{
		bool result;
		result = (m_opcode==x?true:false);
		return result;
	};
	inline bool operator != (mathop::opcode x)
	{
		bool result;
		result = (m_opcode!=x?true:false);
		return result;
	};
};

class EXPRESSION
{
	char *algStr,*polStr;
};

class EQUATIOM
{
	EXPRESSION *expr1, *expr2;
};


class math_object
{
private:
	text_object *result;

private:
	void put_to_polish (char the_token);
	int lex_level, output_count;

public:
	math_object ();
	~math_object ();
	math_object &operator >> (char *(&dest));
	operation detect (char *theToken);
	void push (operation &the);
	FRACTIONS::fraction evaluate (text_object program);
	FRACTIONS::fraction calculate (FRACTIONS::fraction arg1, FRACTIONS::fraction arg2, operation opCode);
	text_object alg2polish (text_object source);
	text_object text2numeric (text_object source);
};
