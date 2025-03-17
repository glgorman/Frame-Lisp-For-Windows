
// template for string types with fixed size
// which is created by usihg a typedef as
// needed

#define IDENTSIZE	31
#define MAXCURSOR	8191
#define MAXCODE		4095

template <size_t sz> class pstring;
typedef pstring<IDENTSIZE+1>	ALPHA;
typedef pstring<MAXCURSOR+256>	SYMBUFARRAY;
typedef pstring<MAXCODE+1>		CODEARRAY;

template <size_t sz>
class pstring
{
protected:
	char str[sz];

public:
	pstring() { memset(str, 0, sz); }
	void* operator new(size_t size, void* ptr);
	pstring &operator=(const char* s);
	char& operator[](int i) { return str[i]; }
	const char& operator[](int i) const { return str[i]; }
	size_t length() const { return strlen(str); }
	operator char* () { return str; }
	operator const char* () const { return str; }
};

template <size_t sz>
void *pstring<sz>::operator new (size_t sz1,void* ptr2)
{
	size_t sz2;
	sz2 = sizeof(pstring<sz>);
	pstring<sz> *ptr;
	if (ptr2==NULL)
	{
		ptr = (pstring<sz>*) malloc (sz2);
	}
	else
		ptr = (pstring<sz>*) ptr2;
	memset (ptr->str,0,sz);
	return (void*) ptr;
}

template <size_t sz>
pstring<sz> &pstring<sz>::operator = (const char* ptr)
{
	if (ptr) {
		strncpy_s(str, sz, ptr, sz - 1);
		str[sz - 1] = '\0';  // Ensure null-termination
	}
	else {
		str[0] = '\0';
	}
	return *this;
}
