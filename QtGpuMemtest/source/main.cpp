#include "qtgpumemtest.h"
#include <QtGui/QApplication>

int main(int argc, char *argv[])
{
	// Set up the application parameters
	QCoreApplication::setOrganizationName("NCSA");
	QCoreApplication::setOrganizationDomain("http://cudagpumemtest.sourceforge.net/");
	QCoreApplication::setApplicationName("CUDA GPU Memtest");

	qRegisterMetaType<TestInfo>("TestInfo");

	QApplication a(argc, argv);
	QtGpuMemtest w;
	w.show();
	return a.exec();
}
