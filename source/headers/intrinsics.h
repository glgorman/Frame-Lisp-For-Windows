//
// Intrinsic functions needed to support Pascal like
// languages.  Copyright 2021 Gerold Lee Gorman
// Permission to redistribute and make use of this
// software under GNU/MIT license.
//
#include <vector>
#include "pstring.h"
#include <wchar.h>

using namespace std;

#define KEYBOARD (1)
#define output	 (1)
#define CONSOLE  (2)
#define LOCK   (false)
#ifndef YES
#define YES		(1)
#endif
#define HAS_CONSOLE		(YES)

typedef FILE PHYLE;
typedef	enum { CHAR1, CHARPTR1, DOUBLE1, DWORD1, FLOAT1, HEXCH, INT1, SIZE1, ULONG1, VOID1 } PTYPE;

/*NAMES*/
typedef enum
{
	NONE,
	TYPES,
	KONST,
	FORMALVARS,
	ACTUALVARS,
	FIELD,
	PROC1,
	FUNC,
	MODULE,
} IDCLASS;

typedef enum
{
	UNDEFINED,
	SCALAR,
	SUBRANGE,
	POINTER,
	LONGINT,
	POWER,
	ARRAYS,
	RECORDS,
	FILES,
	TAGFLD,
	VARIANT2,
} STRUCTFORM;

class sandbox
{
	void	*m_buffer;
	size_t	m_size;
	size_t	m_pos;

public:
	sandbox ();
	static void *allocate_p(size_t size);
	void *pascal_new (size_t sz);
	void pascal_mark (void *(& ptr));
	void pascal_release (void *(& ptr));
};

struct pascal_error
{
	char *errstr;
	int	 errnum;

	pascal_error()
	{
		errstr = nullptr;
		errnum = -1;
	}
	pascal_error (int id, char *str)
	{
		errnum = id;
		errstr = str;
	}
};

extern pascal_error error_list[];

class pascal_file
{
public:
	int blocks_written;
	int blocks_read;
	vector<char*> *m_source;
	vector<char*>::iterator m_begin;
	vector<char*>::iterator m_pos;	
	char *_tmpfname;

public:
	pascal_file ();
	size_t size ();
	void append (char *);
	char *get_sector (int n);
	void *write_sector (int n, char *);
};

class EXIT_CODE
{
public:
	bool m_edit;
	int	 err;
	char *m_str;

public:
	EXIT_CODE(char*str);
	EXIT_CODE(int n, bool edit);
};

class s_param
{
public:
	PTYPE	m_type;
	union
	{
		char	ch;
		char	*str;
		double	d;
		float	f;
		int		i;
		size_t	sz;
	};

	s_param(unsigned long arg);
	s_param (char arg);
	s_param (char* arg);
	s_param (const char* arg);
	s_param (ALPHA &arg);
	s_param (double arg);
	s_param (float arg);
	s_param (int arg);
	s_param (size_t arg);
	// warning - unsafe
	s_param (void *arg);
};

namespace pascal0
{
struct key_info;
};

class identifier;
typedef identifier* CTP;
class structure;
typedef structure* STP;

namespace treetype
{
	pascal0::key_info *get_key_info (int index);
	void struct_info (const CTP node, STP stp);
	void symbol_dump(const CTP &n1, int i, IDCLASS ftype);
	int compare (ALPHA &str1, ALPHA &str2);
	int idsearch (const CTP& n1, CTP& n2, ALPHA &str);
	int keysearch (int pos, char *&str);
	void printleaf (const CTP node, IDCLASS target);
	void printtree (char *tag, const CTP &n1, int i, IDCLASS, bool verbose);
	void printtree1 (const CTP &n1);
	void reset_symbols ();
	void TRAP1 (char *str1, char *str2);
	void TRAP2 (char *str1, char *str2, CTP LCP0);
	void TRAP3 (char *str1, char *str2);
};

namespace SYSCOMM
{
using namespace std;

	void LAUNCH_CONSOLE();
	void LAUNCH_CONSOLE(const char *);
	void REWRITE(pascal_file*,char*);
	void RESET(pascal_file*,char*);
	bool IORESULT(void);
	void OPENNEW(pascal_file *,char *);
	void OPENOLD(pascal_file *,char *);
	void READ(int, char &);
	int CLOSE(pascal_file*,bool lock=false);
	int UNITWRITE (int UNITNUMBER, char *ARRAY, int LENGTH, int BLOCK=0, unsigned int MODE=0);
	int BLOCKWRITE(pascal_file *, const unsigned char *param2, int param3,int param4=0);
	int BLOCKREAD(pascal_file *, char *param2, int param3,int &param4);
	void OutputDebugString (const char *str);
};

void READLN(int uid, char *(&));

void _WRITE(int uid, bool, size_t argc, ...);
void write(int uid, const s_param &);
void write(int uid, const s_param &,const s_param &);
void write(int uid, const s_param &,const s_param &,const s_param &);
void write(int uid, const s_param &,const s_param &,const s_param &,const s_param &);
void write(int uid, const s_param &,const s_param &,const s_param &,const s_param &,const s_param &);
void write(int uid, const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &);
void write(int uid, const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &);
void write(int uid, const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &);
void write(int uid, const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,
		   const s_param &,const s_param &,const s_param &);
void write(int uid, const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,
		   const s_param &,const s_param &,const s_param &,const s_param &);
void write(int uid, const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,
		   const s_param &,const s_param &,const s_param &,const s_param &,const s_param &);
void write(int uid, const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,
		   const s_param &,const s_param &,const s_param &,const s_param &,const s_param &,const s_param &);

void writeln(int uid);
void writeln(int uid, const s_param &);
void writeln(int uid, const s_param &,const s_param &);
void writeln(int uid, const s_param &,const s_param &,const s_param &);
void writeln(int uid, const s_param &,const s_param &,const s_param &,const s_param &);
void writeln(int uid, const s_param &,const s_param &,const s_param &,const s_param &,
			 const s_param &);
void writeln(int uid, const s_param &,const s_param &,const s_param &,const s_param &
			 ,const s_param &,const s_param &);
void writeln(int uid, const s_param &,const s_param &,const s_param &,const s_param &,
			 const s_param &,const s_param &,const s_param &);
void writeln(int uid, const s_param &,const s_param &,const s_param &,const s_param &,
			 const s_param &,const s_param &,const s_param &,const s_param &);


int SCAN(int,bool,char,const char *(a));

size_t MEMAVAIL();
void NEW(void *(&ptr), int sz);
void MARK(void*(&));
void RELEASE(void*);

float PWROFTEN(int i);
int TRUNC(double);
int ORD(const int &);
bool ODD(const int &);
int ROUND (double arg);
void TIME(int &, int &);
void MOVELEFT(const char*,char*,int);
