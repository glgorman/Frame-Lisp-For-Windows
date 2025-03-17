
#include "stdafx.h"
#include "afxmt.h"
#include "new.h"
#include <memory>
#include <vector>
#include <fstream>
#include "defines.h"

using namespace std;

//nclude "symbol_table.h"
#include "bTreeType.h"

//	simple string comparison called from
//	bTreeNode routines -- throws an exception
//	if either argument is NULL;

int btext::compare (char *item1, char *item2)
{
	char buffer1 [256], buffer2 [256];
	int result;
	try {
		if ((item1==NULL)||(item2==NULL))
			throw ((char*)NULL);
		strcpy_s (buffer1,item1);
		strcpy_s (buffer2,item2);
		btext::to_lower (buffer1);
		btext::to_lower (buffer2);
		result = strcmp (buffer1,buffer2);
		return result;
		}
	catch (char* item) {
//		cout << "Null error in bTreeTypes comparison function.\n";
		throw item;
	}
}

void test_btree_allocate ()
{
	size_t sz;
	void *ptr0;
	sz = sizeof(bTreeType<char*>);
	ptr0 =  malloc(sz);
	bTreeType<char*> *str1;
	str1 = new (ptr0)bTreeType<char*>;
}







