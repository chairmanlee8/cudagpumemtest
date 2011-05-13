#ifndef QTGPUMEMTEST_H
#define QTGPUMEMTEST_H

#include <QtGui>
#include <QVector>
#include <QMap>
#include <QFile>
#include <QIODevice>
#include <QSettings>

#include "gpudisplaywidget.h"
#include "ui_qtgpumemtest.h"

#include <cuda_runtime_api.h>

#include "gputests.h"

class QtGpuMemtest : public QMainWindow
{
	Q_OBJECT

public:
	QtGpuMemtest(QWidget *parent = 0, Qt::WFlags flags = 0);
	~QtGpuMemtest();

	static enum ViewMode { BasicView, AdvancedView, BasicResultsView, AdvancedResultsView, NoDevicesView };
	static enum ProgressMode { NoProgress, AdvancedProgress, QuickProgress, StressProgress };

public slots:
	void customStressValue(int minutes);

	void setView(ViewMode viewMode);
	void setProgress(ProgressMode progressMode);

	// some menu and toolbar functions
	void copyResults();
	void exportResults();
	void exit();
	void about();
	void clearDevices();
	void relistDevices();
	void switchView();
	void checkAllDevices(int checked);
	void startChecked();
	void stopAll();
	void returnHome() { setView(BasicView); };

	// Aggregate test options
	void quickTest();
	void quickTestEnded();
	/*void stressTest();*/

	void widgetStartTests(int infinite);
	void widgetStopTests();
	void widgetTestsEnded();

	void progressIncrement();

	//void widgetDisplayResults();

private:
	Ui::QtGpuMemtestClass ui;

	ViewMode					currentViewMode;
	ProgressMode				currentProgressMode;

	QList<GpuDisplayWidget*>	deviceWidgets;
	QList<cudaDeviceProp*>		devices;
	QMap<int, QtGpuThread*>		testThreads;
	QVector<TestInfo>			tests;				// default tests (default template)

	QTimer*						pollTimer;			// 1 second poll timer for use in progress checking
	long						elapsedTime;
};

#endif // QTGPUMEMTEST_H
