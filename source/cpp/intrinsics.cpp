
#include "stdafx.h"
#include <vector>
#include "intrinsics.h"

#define PAGESZ (4096)
#define	BLOCKSIZE	(512)

namespace SYSCOMM
{
	BOOL WINAPI ConsoleHandler(DWORD);
}

pascal_error error_list [] =
{
	pascal_error(1,"Error in simple type"), 
	pascal_error(2,"Identifier expected"),
	pascal_error(3,"'PROGRAM' expected"),
	pascal_error(4,"')' expected"),
	pascal_error(5,": expected"),
	pascal_error(6,"Illegal symbol (possibly missing ';' on line above)"), 
	pascal_error(7,"Error in parameter list"), 
	pascal_error(8,"'OF' expected"), 
	pascal_error(9,"'(' expected"), 
	pascal_error(10,"Error in type"), 
	pascal_error(11,"[ expected"), 
	pascal_error(12,"']'expected"), 
	pascal_error(13,"'END' expected"), 
	pascal_error(14,"';' expected (possibly on line above)"), 
	pascal_error(15,"INTEGER expected"), 
	pascal_error(16,"= expected"), 
	pascal_error(17,"'BEGIN' expected"),
	pascal_error(18,"Error in declaration part"), 
	pascal_error(19,"Error in <field-list>"), 
	pascal_error(20,"'.'expected"), 
	pascal_error(21,"'*' expected"), 
	pascal_error(22,"'Interface' expected"), 
	pascal_error(23,"'Implementation' expected"), 
	pascal_error(24,"'Unit' expected"), 
	pascal_error(50,"Error in constant"), 
	pascal_error(51,"'=' expected"), 
	pascal_error(52,"'THEN' expected"), 
	pascal_error(53,"'UNTIL' expected"),
	pascal_error(54,"'DO' expected"), 
	pascal_error(55,"'TO' or 'DOWNTO' expected in for statement"), 
	pascal_error(56,"'IF' expected"), 
	pascal_error(57,"'FILE' expected"), 
	pascal_error(58,"Error in <factor> (bad expression)"), 
	pascal_error(59,"Error in variable"), 
	pascal_error(101,"Identifier declared twice"),  
	pascal_error(102,"Low bound exceeds high bound"), 
	pascal_error(103,"Identifier is not of the appropriate class"), 
	pascal_error(104,"Undeclared identifier"), 
	pascal_error(105,"Sign not allowed"), 
	pascal_error(106,"Number expected"), 
	pascal_error(107,"Incompatible subrange types"), 
	pascal_error(108,"File not allowed here"), 
	pascal_error(109,"Type must not be real"), 
	pascal_error(110,"<tagfield> type must be scalar or subrange"), 
	pascal_error(111,"Incompatible with <tagfield> part"), 
	pascal_error(112,"Index type must not be real"), 
	pascal_error(113,"Index type must be a scalar or a subrange"), 
	pascal_error(114,"Base type must not be real"), 
	pascal_error(115,"Base type must be a scalar or a subrange"), 
	pascal_error(116,"Error in type of standard procedure parameter"), 
	pascal_error(117,"Unsatisfied forward reference"), 
	pascal_error(118,"Forward reference type identifier in variable declaration"), 
	pascal_error(119,"Re-specified parameters not OK for a forward declared procedure"), 
	pascal_error(120,"Function result type must be scalar, subrange or pointer"), 
	pascal_error(121,"File value parameter not allowed"), 
	pascal_error(122,"A forward declared function's result type can't be re-specified"), 
	pascal_error(123,"Missing result type in function declaration"), 
	pascal_error(124,"F-format for reals only"),
	pascal_error(125,"Error in type of standard procedure parameter"), 
	pascal_error(126,"Number of parameters does not agree with declaration"), 
	pascal_error(127,"Illegal parameter substitution"), 
	pascal_error(128,"Result type does not agree with declaration"), 
	pascal_error(129,"Type conflict of operands"), 
	pascal_error(130,"Expression is not of set type"), 
	pascal_error(131,"Tests on equality allowed only"), 
	pascal_error(132,"Strict inclusion not allowed"), 
	pascal_error(133,"File comparison not allowed"), 
	pascal_error(134,"Illegal type of operand(s)"), 
	pascal_error(135,"Type of operand must be boolean"), 
	pascal_error(136,"Set element type must be scalar or subrange"), 
	pascal_error(137,"Set element types must be compatible"), 
	pascal_error(138,"Type of variable is not array"), 
	pascal_error(139,"Index type is not compatible with the declaration"), 
	pascal_error(140,"Type of variable is not record"), 
	pascal_error(141,"Type of variable must be file or pointer"), 
	pascal_error(142,"Illegal parameter solution"), 
	pascal_error(143,"Illegal type of loop control variable"), 
	pascal_error(144,"Illegal type of expression"), 
	pascal_error(145,"Type conflict"), 
	pascal_error(146,"Assignment of files not allowed"), 
	pascal_error(147,"Label type incompatible with selecting expression"), 
	pascal_error(148,"Subrange bounds must be scalar"), 
	pascal_error(149,"Index type must be integer"), 
	pascal_error(150,"Assignment to standard function is not allowed"), 
	pascal_error(151,"Assignment to formal function is not allowed"), 
	pascal_error(152,"No such field in this record"), 
	pascal_error(153,"Type error in read"), 
	pascal_error(154,"Actual parameter must be a variable"), 
	pascal_error(155,"Control variable cannot be formal or non-local"), 
	pascal_error(156,"Multidefined case label"), 
	pascal_error(157,"Too many cases in case statement"), 
	pascal_error(158,"No such variant in this record"), 
	pascal_error(159,"Real or string tagfields not allowed"), 
	pascal_error(160,"Previous declaration was not forward"), 
	pascal_error(161,"Again forward declared"), 
	pascal_error(162,"Parameter size must be constant"), 
	pascal_error(163,"Missing variant in declaration"), 
	pascal_error(164,"Substitution of standard proc/func not allowed"), 
	pascal_error(165,"Multidefined label"), 
	pascal_error(166,"Multideclared label"), 
	pascal_error(167,"Undeclared label"), 
	pascal_error(168,"Undefined label"), 
	pascal_error(169,"Error in base set"), 
	pascal_error(170,"Value parameter expected"), 
	pascal_error(171,"Standard file was re-declared"), 
	pascal_error(172,"Undeclared external file"), 
	pascal_error(174,"Pascal function or procedure expected"), 
	pascal_error(182,"Nested units not allowed"), 
	pascal_error(183,"External declaration not allowed at this nesting level"), 
	pascal_error(184,"External declaration not allowed in interface section"), 
	pascal_error(185,"Segment declaration not allowed in unit"), 
	pascal_error(186,"Labels not allowed in interface section"), 
	pascal_error(187,"Attempt to open library unsuccessful"), 
	pascal_error(188,"Unit not declared in previous 'Uses' declaration"), 
	pascal_error(189,"'Uses' not allowed at this nesting level"), 
	pascal_error(190,"Unit not in library"), 
	pascal_error(191,"No private files"), 
	pascal_error(192,"'Uses' must be in interface section"),
	pascal_error(193,"Not enough room for this operation"), 
	pascal_error(194,"Comment must appear at top of program"), 
	pascal_error(195,"Unit not importable"), 
	pascal_error(201,"Error in real numbe.r - digit expected"), 
	pascal_error(202,"String constant must not exceed source line"), 
	pascal_error(203,"Integer constant exceeds range"), 
	pascal_error(204,"8 or 9 in octal number"), 
	pascal_error(250,"Too many scopes of nested identifiers"), 
	pascal_error(251,"Too many nested procedures or functions"), 
	pascal_error(252,"Too many forward references of procedure entries"), 
	pascal_error(253,"Procedure too long"),
	pascal_error(254,"Too many long constants in this procedure"), 
	pascal_error(256,"Too many external references"), 
	pascal_error(257,"Too many externals"), 
	pascal_error(258,"Too many local files"), 
	pascal_error(259,"Expression too complicated"), 
	pascal_error(300,"Division by zero"), 
	pascal_error(301,"No case provided for this value"), 
	pascal_error(302,"Index expression out of bounds"), 
	pascal_error(303,"Value to be assigned is out of bounds"), 
	pascal_error(304,"Element expression out of range"),  
	pascal_error(350,"No data segment allocated"), 
	pascal_error(351,"Segment used twice"), 
	pascal_error(352,"No code segment allocated"), 
	pascal_error(353,"Non-intrinsic unit called from intrinsic unit"), 
	pascal_error(354,"Too many segments for the segment dictionary"), 
	pascal_error(398,"Implementation restriction"), 
	pascal_error(399,"Implementation restriction"), 
	pascal_error(400,"Illegal character in text"), 
	pascal_error(401,"Unexpected end of input"), 
	pascal_error(402,"Error in writing code file, not enough room"), 
	pascal_error(403,"Error in reading include file"),
	pascal_error(404,"Error in writing list file, not enough room"), 
	pascal_error(405,"Call not allowed in separate procedure"), 
	pascal_error(406,"Include file not legal"), 
	pascal_error(407,"Too many libraries"),
};

template <typename X>
class variadic 
{
private:
	size_t sz;
	va_list vl;
	X *ptr1;
	vector<X> ptr2;
	X *get()
	{
		X *result;
		result=va_arg(vl,const X*);
		return result;
	}

public:
	variadic (size_t sz,...)
	{
		va_start(vl,sz);
		ptr2.resize (sz);
	}
	void construct ()
	{
		int i;
		for (i=0;i<sz;i++)
			ptr2[i]=get();
	}
};

class slist: public variadic<s_param>
{

};

variadic<int> *doubler (size_t sz,...)
{
	variadic<int> *result;
	result = NULL;
	return result;
}

void test_variadic ()
{
	variadic<int> *data;
	data = doubler (1,2,3,4,5);
}

pascal_file::pascal_file ()
{
	blocks_written = 0;
	blocks_read = 0;
	_tmpfname = "";
	m_source = new vector<char*>;
	m_source->reserve (2048);
}

size_t pascal_file::size ()
{
	size_t result;
	result = m_source->size();
	return result;
}

void pascal_file::append (char *str)
{
	size_t sz = m_source->size();
	m_source->resize(sz+1);
	m_source->at (sz) = str;
}

char *pascal_file::get_sector (int n)
{
	char *str;
	size_t sz = m_source->size();
	if (n>=sz)
		return NULL;
	str = m_source->at (n);
	return str;
}

EXIT_CODE::EXIT_CODE(char*str)
{
	m_edit = false;
	m_str = str;
	err = 0;
}
	
EXIT_CODE::EXIT_CODE(int n, bool edit)
{
	m_edit = edit;
	int j = 0;
	err = n;
	m_str = NULL;
	while (error_list[j].errstr!=NULL)
	{
		if (n==error_list[j].errnum)
		{
			m_str = error_list[j].errstr;
			break;
		}
		j++;
	}
}

// 16MB sounds good for now
// TODO make something reasonable

float PWROFTEN(int arg)
{
	float result = 1;
	int i;
	for (i=0;i<arg;i++)
		result*=10;
	return result;
}

void MARK(void* &PTR)
{
	PTR = 0;
}

void RELEASE(void *ptr)
{
	delete ptr;
}

size_t MEMAVAIL()
{
	size_t result;
	result = 1<<24;
	return result;
}

void MOVELEFT(const char *src, char *dest, int count)
{
	memcpy(dest,src,count);
}

int SCAN(int max,bool find,char ch, const char *str)
{
	int i;
	char ch1;
	int pos = 0;
	for (i=0;i<max;i++)
	{
		ch1= str[i];
		if ((ch1=ch)&&(find==true))
		{
			pos=i;
			break;
		}
		else
		if ((ch1!=ch)&&(find==false))
		{
			pos=i;
			break;
		}
	}
	return pos;
}

int ROUND (double arg)
{
	int result;
	result = (int)(arg+0.5000);
	return result;
}

int TRUNC(double d)
{
	int result;
	result = (int)d;
	return result;
}

int ORD(const int &arg)
{
	return arg;
}

bool ODD(const int &arg)
{
	bool result;
	if (arg&0x01)
		result = true;
	else
		result = false;
	return result;
}

void TIME(int &tick1, int &tick2)
{
	tick2 = GetTickCount();
	tick1 = tick2/1000;;
}

s_param::s_param(char arg)
{
	m_type = CHAR1;
	ch = arg;
}
s_param::s_param(char* arg)
{
	m_type = CHARPTR1;
	str = arg;
}
s_param::s_param(const char* arg)
{
	m_type = CHARPTR1;
	str = (char*)&arg[0];
}
s_param::s_param(ALPHA &arg)
{
	m_type = CHARPTR1;
	str = (char*)arg;
}
s_param::s_param(unsigned long arg)
{
	m_type = ULONG1;
	ch = arg;
}
s_param::s_param(double arg)
{
	m_type = DOUBLE1;
	d = arg;
}
s_param::s_param(float arg)
{
	m_type = FLOAT1;
	f = arg;
}
s_param::s_param(int arg)
{
	m_type = INT1;
	i = arg;
}
s_param::s_param(LPVOID arg)
{
	m_type = VOID1;
	i = (unsigned int)arg;
}
s_param::s_param(size_t arg)
{
	m_type = SIZE1;
	sz = arg;
}

void write(int uid, const s_param &w1)
{
	_WRITE(uid,false,1,&w1);
}

void write(int uid, const s_param &w1, const s_param &w2)
{
	_WRITE(uid,false,2,&w1,&w2);
}

void write(int uid, const s_param &w1, const s_param &w2, const s_param &w3)
{
	_WRITE(uid,false,3,&w1,&w2,&w3);
}

void write(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4)
{
	_WRITE(uid,false,4,&w1,&w2,&w3,&w4);
}

void write(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
	const s_param &w5)
{
	_WRITE(uid,false,5,&w1,&w2,&w3,&w4,&w5);
}

void write(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
	const s_param &w5, const s_param &w6)
{
	_WRITE(uid,false,6,&w1,&w2,&w3,&w4,&w5,&w6);
}

void write(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
	const s_param &w5, const s_param &w6, const s_param &w7)
{
	_WRITE(uid,false,7,&w1,&w2,&w3,&w4,&w5,&w6,&w7);
}

void write(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
	const s_param &w5, const s_param &w6, const s_param &w7, const s_param &w8)
{
	_WRITE(uid,false,8,&w1,&w2,&w3,&w4,&w5,&w6,&w7,&w8);
}

void write(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
	const s_param &w5, const s_param &w6, const s_param &w7, const s_param &w8, const s_param &w9)
{
	_WRITE(uid,false,9,&w1,&w2,&w3,&w4,&w5,&w6,&w7,&w8,&w9);
}

void write(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
	const s_param &w5, const s_param &w6, const s_param &w7, const s_param &w8,
	const s_param &w9, const s_param &w10)
{
	_WRITE(uid,false,10,&w1,&w2,&w3,&w4,&w5,&w6,&w7,&w8,&w9,&w10);
}

void write(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
	const s_param &w5, const s_param &w6, const s_param &w7, const s_param &w8,
	const s_param &w9, const s_param &w10, const s_param &w11)
{
	_WRITE(uid,false,11,&w1,&w2,&w3,&w4,&w5,&w6,&w7,&w8,&w9,&w10,&w11);
}

void write(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
	const s_param &w5, const s_param &w6, const s_param &w7, const s_param &w8,
	const s_param &w9, const s_param &w10, const s_param &w11, const s_param &w12)
{
	_WRITE(uid,false,12,&w1,&w2,&w3,&w4,&w5,&w6,&w7,&w8,&w9,&w10,&w11,&w12);
}

void writeln(int uid)
{
	_WRITE(uid,true,0,NULL);
}

void writeln(int uid, const s_param &w1)
{
	_WRITE(uid,true,1,&w1);
}

void writeln(int uid, const s_param &w1, const s_param &w2)
{
	_WRITE(uid,true,2,&w1,&w2);
}

void writeln(int uid, const s_param &w1, const s_param &w2, const s_param &w3)
{
	_WRITE(uid,true,3,&w1,&w2,&w3);
}

void writeln(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4)
{
	_WRITE(uid,true,4,&w1,&w2,&w3,&w4);
}

void writeln(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
		   const s_param &w5)
{
	_WRITE(uid,true,5,&w1,&w2,&w3,&w4,&w5);
}

void writeln(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
		   const s_param &w5, const s_param &w6)
{
	_WRITE(uid,true,6,&w1,&w2,&w3,&w4,&w5,&w6);
}

void writeln(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
		   const s_param &w5, const s_param &w6, const s_param &w7)
{
	_WRITE(uid,true,7,&w1,&w2,&w3,&w4,&w5,&w6,&w7);
}

void writeln(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
		   const s_param &w5, const s_param &w6, const s_param &w7, const s_param &w8)
{
	_WRITE(uid,true,8,&w1,&w2,&w3,&w4,&w5,&w6,&w7,&w8);
}

void writeln(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
		   const s_param &w5, const s_param &w6, const s_param &w7, const s_param &w8, const s_param &w9)
{
	_WRITE(uid,true,9,&w1,&w2,&w3,&w4,&w5,&w6,&w7,&w8,&w9);
}

void writeln(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
		   const s_param &w5, const s_param &w6, const s_param &w7, const s_param &w8,
		   const s_param &w9, const s_param &w10)
{
	_WRITE(uid,true,10,&w1,&w2,&w3,&w4,&w5,&w6,&w7,&w8,&w9,&w10);
}

void writeln(int uid, const s_param &w1, const s_param &w2, const s_param &w3,const s_param &w4,
		   const s_param &w5, const s_param &w6, const s_param &w7, const s_param &w8,
		   const s_param &w9, const s_param &w10, const s_param &w11)
{
	_WRITE(uid,true,11,&w1,&w2,&w3,&w4,&w5,&w6,&w7,&w8,&w9,&w10,&w11);
}

void writeln(int uid, const s_param &w1, const s_param &w2, const s_param &w3, const s_param &w4,
		   const s_param &w5, const s_param &w6, const s_param &w7, const s_param &w8,
		   const s_param &w9, const s_param &w10, const s_param &w11, const s_param &w12)
{
	_WRITE(uid,true,12,&w1,&w2,&w3,&w4,&w5,&w6,&w7,&w8,&w9,&w10,&w11,&w12);
}

void _WRITE(int uid, bool crlf, size_t sz,...)
{
	size_t width;
	char *ptr;
	char buffer1[PAGESZ];
	char buffer2[PAGESZ];
	const s_param *val;
	unsigned int i;
	va_list vl;
	va_start(vl,sz);
	i=0;
	memset(buffer1,0,PAGESZ);
	memset(buffer2,0,PAGESZ);
	while(i<sz)
	{
		val=va_arg(vl,const s_param *);
		switch (val->m_type)
		{
		case CHAR1:
			sprintf_s(buffer1,PAGESZ,"%c",val->ch);
			break;
		case CHARPTR1:
			if (val->str==NULL)
				ptr = "<null>";
			else
				ptr = val->str;
			sprintf_s(buffer1,PAGESZ,"%s",ptr);
			break;
		case DOUBLE1:
			sprintf_s(buffer1,PAGESZ,"%lf",val->d);
			break;
		case FLOAT1:
			sprintf_s(buffer1,PAGESZ,"%f",val->f);
			break;
		case HEXCH:
			sprintf_s(buffer1,PAGESZ,"%08x",val->i);
			break;
		case INT1:
			sprintf_s(buffer1,PAGESZ,"%d",val->i);
			break;
		case SIZE1:
			width = val->sz;
			sprintf_s(buffer1,PAGESZ,"%s"," ");
			break;
		case VOID1:
			sprintf_s(buffer1,PAGESZ,"%08x",val->i);
			break;
		default:
			break;
		}
		strcat_s(buffer2,PAGESZ,buffer1);
		i++;
	}
	if (crlf==true)
		strcat_s(buffer2,PAGESZ,"\n");
	SYSCOMM::OutputDebugString(buffer2);
	va_end(vl);
}

void SYSCOMM::LAUNCH_CONSOLE(const char *str)
{
#ifdef HAS_CONSOLE
	AllocConsole();
	SetConsoleCtrlHandler(ConsoleHandler, TRUE);
	writeln(output, str);
#endif
}

void SYSCOMM::LAUNCH_CONSOLE()
{
#ifdef HAS_CONSOLE
	AllocConsole();
	SetConsoleCtrlHandler(ConsoleHandler,TRUE);
#endif
}

BOOL WINAPI SYSCOMM::ConsoleHandler(DWORD dwEvent)
{
#if 0
	Beep(262,300);
#endif
	if (dwEvent==CTRL_CLOSE_EVENT)
		return true;
	else
		return false;
}

void SYSCOMM::OutputDebugString (const char *str)
{
	WPARAM w;
	LPARAM l;
	CWinApp *pApp;
	pApp = AfxGetApp();
//	CMainFrame *pMain = (CMainFrame*)pApp->m_pMainWnd;
#if 0
//	TODO - put in a system for various debug levels
//	like typedef enum { NONE, LACONIC, MEDIUM, VERBOSE, }
//	debug_level;
	if (m_debug==false)
		return;
#endif
	w = (WPARAM) _strdup(str);
	l = 0;
#ifdef HAS_CONSOLE
	DWORD dwWritten;
	HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	WriteFile(hOut,(LPVOID)str,(DWORD)strlen(str),&dwWritten,NULL);
#endif
//	if (pMain!=NULL)
//	{
//	  pMain->PostMessage(WM_RECIEVE_DEBUG_DATA,w,l);
//	}
//	else
		delete (char*)w;
//	Sleep(1);
}

void SYSCOMM::OPENNEW(pascal_file *file,char *)
{
	vector<char*> *source = file->m_source;
	vector<char*>::iterator &iter = source->begin();
	file->m_pos = iter;
	file->blocks_written = 0;
}

void SYSCOMM::REWRITE(pascal_file *,char *)
{
	ASSERT(false);
}

void SYSCOMM::RESET(pascal_file *file,char*)
{
	vector<char*> *source = file->m_source;
	vector<char*>::iterator &iter = source->begin();
	file->m_begin = iter;
	file->m_pos = iter;
	file->blocks_read = 0;
	file->blocks_written = 0;
}

int SYSCOMM::CLOSE(pascal_file *f,bool)
{
// 	ASSERT(0);
	return 0;
}

void READLN(int id, char *(&str))
{
	BOOL bres;
	char ptr[4];
	int n = 0;
	DWORD	sz,mode;
	HANDLE hIn = GetStdHandle(STD_INPUT_HANDLE);
	GetConsoleMode(hIn,&mode);
//	mode &= !ENABLE_LINE_INPUT;
	mode = ENABLE_ECHO_INPUT|ENABLE_LINE_INPUT;
	SetConsoleMode(hIn,mode);
	while (true)
	{
		do
		{
			bres = ReadConsole(hIn,ptr,1,&sz,NULL);
		}
		while (sz==0);
		str[n]=ptr[0];
		n++;
		if (ptr[0]==13)
			break;
	}
	str[n]=0;
}

void SYSCOMM::READ(int id, char &a)
{
	BOOL bres;
	char buffer[64];
	DWORD	sz,mode;
	HANDLE hIn = GetStdHandle(STD_INPUT_HANDLE);
	GetConsoleMode(hIn,&mode);
	mode &= (!ENABLE_LINE_INPUT)|ENABLE_ECHO_INPUT;
	SetConsoleMode(hIn,mode);
	do
	{
		bres = ReadConsole(hIn,buffer,1,&sz,NULL);
	}
	while (sz==0);
	a = buffer[0];
}

bool SYSCOMM::IORESULT(void)
{
	return true;
}

void SYSCOMM::OPENOLD(pascal_file *file,char *)
{
	vector<char*> *source = file->m_source;
	vector<char*>::iterator &iter = source->begin();
	file->blocks_read = 0;
	file->blocks_written = 0;
	file->m_pos = iter;
}

int SYSCOMM::UNITWRITE (int UNITNUMBER, char *ARRAY, int LENGTH, int BLOCK, unsigned int MODE)
{
	int result = -1;
	switch (UNITNUMBER)
	{
	case 3: // Terminal
		if (MODE==0) {
			writeln(output,ARRAY);
		}
		result = 0;
		break;

	default:
		break;
	}
	return result;
}

// a pascal block is 512 bytes, but an Apple II
// sector is 256 bytes.  m_source is based on
// 256 byte sector style granularity.

int SYSCOMM::BLOCKREAD(pascal_file *file, char *buf, int blocks, int &read)
{
#if 0
	writeln(output,"SYSCOMM::BLOCKREAD(FILE*, char*, int, &int)");
#endif

	char* block_ptr = nullptr;
	read = blocks;
	int i;
	int j;
	int result = 0;
	size_t sz = file->size();
	vector<char*>::iterator &iter = file->m_pos;
	vector<char*>::iterator &end = file->m_source->end();
	if (iter==file->m_source->end())
	{
		read=0;
		return 0;
	}
	memset(buf,0,BLOCKSIZE*blocks);
	int block_pos = file->blocks_read;
	for (i=0;i<blocks;i++)
	{
		for (j=0;j<2;j++)
		{
			if (iter==end)
				break;
			block_ptr = (*iter++);
			if (block_ptr==NULL)
				break;
			memcpy(&(buf[BLOCKSIZE*i+(256*j)]),block_ptr,256);
			buf[BLOCKSIZE*i+256*(j+1)]=0;
		}
		if (block_ptr!=NULL)
		{
			file->blocks_read++;
			result++;		
		}
	}
//	writeln(output,"BLOCK ",block_pos,"\n>>>>>>>>",buf,"<<<<<<<<");
	return result;
}

int SYSCOMM::BLOCKWRITE(pascal_file *file, const unsigned char *buf, int blocks, int offset)
{
	writeln (output,"SYSCOMM::BLOCKWRITE offset= ",offset);
//	assume that we wrote a blcok
//	todo - save the blocks in memory so we can have
//	a hex view - or examine with the dissassembler
	char hexbuf[32];
	char strbuf[32];
	unsigned char ch;
	int h,i,j;
	DWORD k,position;
	char hexchar[] = "0123456789abcdef";
//	for (h=0;h<blocks;h++)
	for (i=0;i<32;i++)
	{
		k = 512*offset+16*i;
		position = file->blocks_written*512+k;
		sprintf_s (hexbuf,32,"%08x: ",position);
		write (output,hexbuf);
		for (j=0;j<16;j++)
		{
			ch = buf[k+j];
			hexbuf[0]=hexchar[(0xf0&ch)>>4];
			hexbuf[1]=hexchar[(0x0f&ch)];
			hexbuf[2]=0;
			hexbuf[3]=0;
			write(output,hexbuf);
		}
		for (j=0;j<16;j++)
		{
			ch = buf[k+j];
			if (ch<32)
				ch+=32;
		// these characters are messing up the
		// screen dump
			if ((ch==0xe0)||(ch==0xe8)
				||(ch==0xe9)||(ch==0xed))
				ch='#';
			strbuf[j]=ch;
		}
		strbuf[j]=0;
		writeln(output," --> \"",strbuf,"\"");
	}
	file->blocks_written++;
	int result = 1;
	return result;
}
