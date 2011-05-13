#include "qtgpumemtest.h"

QtGpuMemtest::QtGpuMemtest(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);

	// Create tests
	tests.push_back(TestInfo(0,		QString("Test0 [Walking 1 bit]"),					QString("0"),		test0,	true));
	tests.push_back(TestInfo(1,		QString("Test1 [Own address tests]"),				QString("1"),		test1,	true));
	tests.push_back(TestInfo(2,		QString("Test2 [Moving inversions, ones&zeros]"),	QString("2"),		test2,	true));
	tests.push_back(TestInfo(3,		QString("Test3 [Moving inversions, 8 bit pat]"),	QString("3"),		test3,	true));
	tests.push_back(TestInfo(4,		QString("Test4 [Moving inversions, random pat]"),	QString("4"),		test4,	true));
	tests.push_back(TestInfo(5,		QString("Test5 [Block move, 64 moves]"),			QString("5"),		test5,	true));
	tests.push_back(TestInfo(6,		QString("Test6 [Moving inversions, 32 bit pat]"),	QString("6"),		test6,	true));
	tests.push_back(TestInfo(7,		QString("Test7 [Random number sequence]"),			QString("7"),		test7,	true));
	tests.push_back(TestInfo(8,		QString("Test8 [Modulo 20, random pattern]"),		QString("8"),		test8,	true));
	tests.push_back(TestInfo(9,		QString("Test9 [Bit fade test]"),					QString("9"),		test9,	false));
	tests.push_back(TestInfo(10,	QString("Test10 [Stress test]"),					QString("Stress"),	test10,	false));

	// Link actions and events
	QSignalMapper* checkMapper = new QSignalMapper(this);
	checkMapper->setMapping(ui.actionCheckAll, 1);
	checkMapper->setMapping(ui.actionCheckNone, 0);
	connect(ui.actionCheckAll, SIGNAL(triggered()), checkMapper, SLOT(map()));
	connect(ui.actionCheckNone, SIGNAL(triggered()), checkMapper, SLOT(map()));
	connect(checkMapper, SIGNAL(mapped(int)), this, SLOT(checkAllDevices(int)));

	// TODO: add some sort of one-time explanation of how to use the advanced mode gui

	connect(ui.actionRelist, SIGNAL(triggered()), this, SLOT(relistDevices()));	
	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(exit()));
	connect(ui.actionAbout, SIGNAL(triggered()), this, SLOT(about()));
	connect(ui.actionStartChecked, SIGNAL(triggered()), this, SLOT(startChecked()));
	connect(ui.actionStopAll, SIGNAL(triggered()), this, SLOT(stopAll()));
	connect(ui.actionClipboardResults, SIGNAL(triggered()), this, SLOT(copyResults()));
	connect(ui.actionExportResults, SIGNAL(triggered()), this, SLOT(exportResults()));
	connect(ui.actionSwitchView, SIGNAL(triggered()), this, SLOT(switchView()));
	connect(ui.customStressDial, SIGNAL(valueChanged(int)), this, SLOT(customStressValue(int)));
	connect(ui.quickTestButton, SIGNAL(clicked()), this, SLOT(quickTest()));
	connect(ui.customStressTestButton, SIGNAL(clicked()), this, SLOT(stressTest()));
	connect(ui.buttonReturn, SIGNAL(clicked()), this, SLOT(returnHome()));

	// Load settings and configure application with settings
	QSettings settings;
	int startWizard = settings.value("startWithWizard", 1).toInt();
	int stressTime = settings.value("stressTime", ui.customStressDial->value()).toInt();
	ui.actionShowWizardOnStartup->setChecked(startWizard == 1);
	customStressValue(stressTime);

	// Setup the current state	
	setView(startWizard ? BasicView : AdvancedView);
	setProgress(NoProgress);

	// List CUDA devices
	ui.actionRelist->trigger();
}

QtGpuMemtest::~QtGpuMemtest()
{
	// Save settings
	QSettings settings;
	settings.setValue("startWithWizard", ui.actionShowWizardOnStartup->isChecked() ? 1 : 0);
	settings.setValue("stressTime", ui.customStressDial->value());

	// Destroy GUI
	clearDevices();
}

void QtGpuMemtest::displayLogMessage(TestInfo t, QString l)
{
	QMessageBox::warning(this, tr("Error"), l);
}

//
// Device and test management
//

void QtGpuMemtest::clearDevices()
{
	deviceWidgets.clear();
	devices.clear();

	// clear the rest of the layout
	QLayoutItem *item;
	while((item = ui.verticalLayoutGpus->takeAt(0)) != 0) delete item;
}

void QtGpuMemtest::relistDevices()
{
	int deviceCount = 0;

	clearDevices();
	cudaGetDeviceCount(&deviceCount);

	if(deviceCount <= 0)
	{
		// No devices found.
		setView(NoDevicesView);
		return;
	}

	for(int i = 0; i < deviceCount; i++)
	{
		cudaDeviceProp *aDevice = new cudaDeviceProp();
		int error = cudaGetDeviceProperties(aDevice, i);

		if(error == cudaSuccess)
		{
			GpuDisplayWidget *aWidget = new GpuDisplayWidget();

			char gpuString[256] = "";
			char gpuMemoryString[256] = "";

			sprintf_s(gpuString, sizeof(gpuString), "GPU #%d - %s", i, aDevice->name);
			sprintf_s(gpuMemoryString, sizeof(gpuMemoryString), "%.1f MiB", aDevice->totalGlobalMem / (float)(1024 * 1024));

			aWidget->setTests(tests);
			aWidget->setGpuName(QString(gpuString));
			aWidget->setGpuMemory(QString(gpuMemoryString));
			aWidget->setFont(QFont("Arial", 14));
			aWidget->setIndex(i);
			aWidget->setMaximumHeight(130);
			aWidget->setMinimumWidth(0);

			// Link up the test controller
			connect(aWidget, SIGNAL(startTests(int)), this, SLOT(widgetStartTests(int)));
			connect(aWidget, SIGNAL(stopTests()), this, SLOT(widgetStopTests()));

			deviceWidgets.append(aWidget);
			devices.append(aDevice);
			ui.verticalLayoutGpus->addWidget(aWidget);
		}
	}

	ui.verticalLayoutGpus->addStretch();
}

void QtGpuMemtest::checkAllDevices(int checked)
{
	for(int i = 0; i < deviceWidgets.count(); i++)
	{
		deviceWidgets[i]->setCheckStart(checked);
	}
}

void QtGpuMemtest::startChecked()
{
	for(int i = 0; i < deviceWidgets.count(); i++)
	{
		deviceWidgets[i]->startChecked();
	}
}

void QtGpuMemtest::stopAll()
{
	for(int i = 0; i < testThreads.count(); i++)
	{
		testThreads[testThreads.keys()[i]]->notifyExit();
	}
}

void QtGpuMemtest::widgetStartTests(int infinite)
{
	int widgetIndex = ((GpuDisplayWidget*)sender())->index();
	QtGpuThread *gpuThread = new QtGpuThread(((GpuDisplayWidget*)sender())->getTests());

	deviceWidgets[widgetIndex]->setCheckStart(true);

	testThreads.remove(widgetIndex);
	testThreads.insert(widgetIndex, gpuThread);

	((GpuDisplayWidget*)sender())->setState(GpuDisplayWidget::RunningMode);
	connect(gpuThread, SIGNAL(testStarting(TestInfo)), deviceWidgets[widgetIndex], SLOT(testStarting(TestInfo)));
	connect(gpuThread, SIGNAL(testFailed(TestInfo)), deviceWidgets[widgetIndex], SLOT(testFailed(TestInfo)));
	connect(gpuThread, SIGNAL(testPassed(TestInfo)), deviceWidgets[widgetIndex], SLOT(testPassed(TestInfo)));
	connect(gpuThread, SIGNAL(log(TestInfo, QString)), this, SLOT(displayLogMessage(TestInfo, QString)));

	// Connect stopped signal
	connect(gpuThread, SIGNAL(finished()), this, SLOT(widgetTestsEnded()));
	connect(gpuThread, SIGNAL(terminated()), this, SLOT(widgetTestsEnded()));

	gpuThread->setDevice(widgetIndex);
	gpuThread->setEndless((bool)infinite);
	gpuThread->start();

	setProgress(AdvancedProgress);
}

void QtGpuMemtest::widgetStopTests()
{
	int widgetIndex = ((GpuDisplayWidget*)sender())->index();
	testThreads[widgetIndex]->notifyExit();
}

void QtGpuMemtest::widgetTestsEnded()
{
	int widgetIndex = ((QtGpuThread*)sender())->deviceIndex();
	/*testThreads.remove(widgetIndex);*/
	deviceWidgets[widgetIndex]->setState(GpuDisplayWidget::StoppedMode);

	bool existsRunning = false;
	for(int i = 0; i < testThreads.count(); i++)
	{
		if(testThreads[testThreads.keys()[i]]->isRunning())
		{
			existsRunning = true;
			break;
		}
	}

	if(!existsRunning)
	{
		setProgress(NoProgress);
		updateResults();
	}
}

//
// Aggregate test options
//

void QtGpuMemtest::quickTest()
{
	QVector<TestInfo> quickTests;

	quickTests.push_back(tests[0]);
	quickTests.push_back(tests[1]);
	quickTests.push_back(tests[2]);
	quickTests.push_back(tests[3]);
	quickTests.push_back(tests[4]);

	// For every device start a thread
	for(int i = 0; i < deviceWidgets.count(); i++)
	{
		QtGpuThread *gpuThread = new QtGpuThread(quickTests);
		gpuThread->setDevice(deviceWidgets[i]->index());

		testThreads.remove(deviceWidgets[i]->index());
		testThreads.insert(deviceWidgets[i]->index(), gpuThread);

		connect(gpuThread, SIGNAL(finished()), this, SLOT(basicTestEnded()));
		connect(gpuThread, SIGNAL(terminated()), this, SLOT(basicTestEnded()));
		connect(gpuThread, SIGNAL(log(TestInfo, QString)), this, SLOT(displayLogMessage(TestInfo, QString)));
		connect(gpuThread, SIGNAL(progressPart()), this, SLOT(progressIncrement()));

		gpuThread->start();
	}

	setProgress(QuickProgress);
}

void QtGpuMemtest::basicTestEnded()
{
	bool existsRunning = false;
	for(int i = 0; i < testThreads.count(); i++)
	{
		if(testThreads[testThreads.keys()[i]]->isRunning())
		{
			existsRunning = true;
			break;
		}
	}

	if(!existsRunning)
	{
		setProgress(NoProgress);
		updateResults();
		setView(BasicResultsView);
	}
}

void QtGpuMemtest::stressTest()
{
	QVector<TestInfo> stressTests;
	stressTests.push_back(tests[10]);

	for(int i = 0; i < deviceWidgets.count(); i++)
	{
		QtGpuThread *gpuThread = new QtGpuThread(stressTests);
		gpuThread->setDevice(deviceWidgets[i]->index());
		gpuThread->setEndless(true);

		testThreads.remove(deviceWidgets[i]->index());
		testThreads.insert(deviceWidgets[i]->index(), gpuThread);

		connect(gpuThread, SIGNAL(log(TestInfo, QString)), this, SLOT(displayLogMessage(TestInfo, QString)));
		connect(gpuThread, SIGNAL(finished()), this, SLOT(basicTestEnded()));
		connect(gpuThread, SIGNAL(terminated()), this, SLOT(basicTestEnded()));

		gpuThread->start();
	}

	pollTimer = new QTimer();
	pollTimer->setInterval(1000);
	pollTimer->setSingleShot(false);
	elapsedTime = 0;

	connect(pollTimer, SIGNAL(timeout()), this, SLOT(progressIncrement()));
	setProgress(StressProgress);

	pollTimer->start();
}

void QtGpuMemtest::stressTestEnded()
{
	pollTimer->stop();
	delete pollTimer;

	for(int i = 0; i < testThreads.count(); i++)
	{
		testThreads[testThreads.keys()[i]]->notifyExit();
	}
}

void QtGpuMemtest::updateResults()
{
	int num_rows = 0;

	for(int i = 0; i < testThreads.count(); i++)
		num_rows += testThreads[testThreads.keys()[i]]->getNumErrors();

	if(num_rows > 0)
	{
		ui.labelPassFail->setText(tr("Failed"));
	}
	else
	{
		ui.labelPassFail->setText(tr("Passed"));
	}

	ui.resultsTable->setRowCount(num_rows);
	ui.resultsTable->setColumnCount(5);

	QStringList header;
	header << "Device" << "Address" << "Expected Value" << "Actual Value" << "Second Read";
	ui.resultsTable->clear();
	ui.resultsTable->setHorizontalHeaderLabels(header);
	ui.resultsTable->setColumnWidth(0, 50);

	int r = 0;
	for(int i = 0; i < testThreads.count(); i++)
	{
		QList<MemoryError> errors = testThreads[testThreads.keys()[i]]->getErrors();
		int k = testThreads[testThreads.keys()[i]]->getNumErrors();

		for(int j = 0; j < k; j++)
		{
			QString base0, base1, base2, base3, base4;
			QTextStream sout0(&base0), sout1(&base1), sout2(&base2), sout3(&base3), sout4(&base4);

			sout0 << testThreads[testThreads.keys()[i]]->deviceIndex();
			ui.resultsTable->setItem(r, 0, new QTableWidgetItem(base0));

			sout1 << "0x" << qSetFieldWidth(8) << qSetPadChar('0') << hex << errors[j].addr;			
			ui.resultsTable->setItem(r, 1, new QTableWidgetItem(base1));

			sout2 << "0x" << qSetFieldWidth(8) << qSetPadChar('0') << hex << errors[j].expected;
			ui.resultsTable->setItem(r, 2, new QTableWidgetItem(base2));

			sout3 << "0x" << qSetFieldWidth(8) << qSetPadChar('0') << hex << errors[j].current;
			ui.resultsTable->setItem(r, 3, new QTableWidgetItem(base3));

			sout4 << "0x" << qSetFieldWidth(8) << qSetPadChar('0') << hex << errors[j].second_read;
			ui.resultsTable->setItem(r, 4, new QTableWidgetItem(base4));

			r++;
		}
	}
}

//
// QtGpuMemtest Slots
//

void QtGpuMemtest::about()
{
	QMessageBox::about(this, QString("CUDA GPU Memtest"),
	                   QString("A GPU memory test utility for NVIDIA and AMD GPUs using well established patterns "
	                           "from memtest86/memtest86+ as well as additional stress tests. The tests are designed "
	                           "to find hardware and soft errors. The code is written in CUDA and OpenCL. "
	                           "\r\n\r\nhttp://cudagpumemtest.sf.net"));
}

void QtGpuMemtest::exit()
{
	this->close();
}

void QtGpuMemtest::customStressValue(int minutes)
{
	QString timeString = "";
	QTextStream tout(&timeString);

	if(minutes >= 60) tout << (minutes / 60) << " Hour ";
	if((minutes % 60) != 0) tout << (minutes % 60) << " Minute ";
	tout << "Stress Burn";

	ui.customStressTestButton->setText(timeString);
	ui.customStressDial->setValue(minutes);
}

//
// State Management
//

void QtGpuMemtest::setView(ViewMode viewMode)
{
	// Switching views should work even when tests are running, assuming that the basic and advanced views are
	// both tied to the same testing model/controller and updated simultaneously.

	currentViewMode = viewMode;

	switch(currentViewMode)
	{
		case BasicView:
			{
				ui.stackedWidget->setCurrentIndex(0);
				ui.actionSwitchView->setText("Advanced View");
				ui.actionSwitchView->setToolTip("Switch to advanced view.");
				ui.actionSwitchView->setEnabled(true);
			}
			break;
		case AdvancedView:
			{
				ui.stackedWidget->setCurrentIndex(1);
				ui.actionSwitchView->setText("Basic View");
				ui.actionSwitchView->setToolTip("Switch to basic wizard.");
				ui.actionSwitchView->setEnabled(true);
			}
			break;
		case BasicResultsView:
			{
				ui.stackedWidget->setCurrentIndex(2);
			}
			break;
		case AdvancedResultsView:
			break;
		case NoDevicesView:
			{
				ui.stackedWidget->setCurrentIndex(3);
				ui.actionSwitchView->setEnabled(false);
			}
			break;
	}

	bool enableAdvancedControls = (currentViewMode == AdvancedView);
	ui.actionCheckAll->setEnabled(enableAdvancedControls);
	ui.actionCheckNone->setEnabled(enableAdvancedControls);
	ui.actionStartChecked->setEnabled(enableAdvancedControls);
	ui.actionStopAll->setEnabled(true);

	if(currentViewMode != BasicView && currentViewMode != AdvancedView)
		ui.actionSwitchView->setEnabled(false);
	else
		ui.actionSwitchView->setEnabled(true);
}

void QtGpuMemtest::setProgress(ProgressMode progressMode)
{
	// If tests are running, disable the relevant controls
	bool predicate = progressMode == NoProgress;

	ui.actionRelist->setEnabled(predicate);
	ui.quickTestButton->setEnabled(predicate);
	ui.customStressTestButton->setEnabled(predicate);
	ui.customStressDial->setEnabled(predicate);
	ui.actionStartChecked->setEnabled(predicate);
	ui.progressBarOverall->setEnabled(predicate);
	ui.actionSwitchView->setEnabled(predicate);

	switch(progressMode)
	{
		case NoProgress:
			{
				ui.progressBarOverall->setRange(0, 1);
				ui.progressBarOverall->setValue(0);
			}
			break;
		case AdvancedProgress:
			{
				ui.progressBarOverall->setRange(0, 0);
				ui.progressBarOverall->reset();
			}
			break;
		case QuickProgress:
			{
				int totalParts = 0;
				for(int i = 0; i < testThreads.count(); i++)
				{
					totalParts += testThreads[testThreads.keys()[i]]->totalProgressParts();
				}

				ui.progressBarOverall->setRange(0, totalParts);
				ui.progressBarOverall->setValue(0);
			}
			break;
		case StressProgress:
			{
				ui.progressBarOverall->setRange(0, 60 * ui.customStressDial->value());
				ui.progressBarOverall->setValue(0);
			}
			break;
	}

	currentProgressMode = progressMode;
}

void QtGpuMemtest::progressIncrement()
{
	if(currentProgressMode == QuickProgress)
	{
		ui.progressBarOverall->setValue(ui.progressBarOverall->value() + 1);
	}
	else if(currentProgressMode == StressProgress)
	{
		if(elapsedTime++ == ui.customStressDial->value() * 60)
			emit stressTestEnded();

		ui.progressBarOverall->setValue(elapsedTime);
	}
}

void QtGpuMemtest::switchView()
{
	// Switching views should work even when tests are running, assuming that the basic and advanced views are
	// both tied to the same testing model/controller and updated simultaneously.
	switch(currentViewMode)
	{
		case BasicView: setView(AdvancedView); break;
		case AdvancedView: setView(BasicView); break;
		case BasicResultsView:
			break;
		case AdvancedResultsView:
			break;
		case NoDevicesView:
			break;
	}
}

void QtGpuMemtest::copyResults()
{
	QString masterOutput;
	QTextStream sout(&masterOutput);

	for(int i = 0; i < testThreads.count(); i++)
	{
		QList<MemoryError> errors = testThreads[testThreads.keys()[i]]->getErrors();
		sout << "Device " << testThreads[testThreads.keys()[i]]->deviceIndex() << " has " << errors.count() << " errors." << endl;

		for(int j = 0; j < errors.count(); j++)
		{
			sout << "Address 0x" << qSetFieldWidth(8) << qSetPadChar('0') << hex << errors[j].addr;
			sout << " expected 0x" << qSetFieldWidth(8) << qSetPadChar('0') << hex << errors[j].expected;
			sout << " returned 0x" << qSetFieldWidth(8) << qSetPadChar('0') << hex << errors[j].current;
			sout.reset();
			sout << endl;
		}
	}

	QClipboard *cb = QApplication::clipboard();
	cb->setText(masterOutput);
}

void QtGpuMemtest::exportResults()
{
	// Select file
	QString fileName = QFileDialog::getSaveFileName(this, tr("Export Results As..."));
	QFile fout(fileName);

	if(fileName == QString::null)
		return;

	if(!fout.open(QIODevice::ReadWrite | QIODevice::Text))
	{
		QMessageBox::warning(this, tr("Error"), tr("Could not export file! Check if file is open elsewhere."));
		return;
	}

	QTextStream sout(&fout);

	for(int i = 0; i < testThreads.count(); i++)
	{
		QList<MemoryError> errors = testThreads[testThreads.keys()[i]]->getErrors();
		sout << "Device " << testThreads[testThreads.keys()[i]]->deviceIndex() << " has " << errors.count() << " errors." << endl;

		for(int j = 0; j < errors.count(); j++)
		{
			sout << "Address 0x" << qSetFieldWidth(8) << qSetPadChar('0') << hex << errors[j].addr;
			sout << " expected 0x" << qSetFieldWidth(8) << qSetPadChar('0') << hex << errors[j].expected;
			sout << " returned 0x" << qSetFieldWidth(8) << qSetPadChar('0') << hex << errors[j].current;
			sout.reset();
			sout << endl;
		}
	}

	fout.close();
}