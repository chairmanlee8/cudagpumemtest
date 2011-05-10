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

	// Setup the current view	
	setView(startWizard ? BasicView : AdvancedView);

	// List CUDA devices
	ui.actionRelist->trigger();

	// Temporary
	allTestsDone = true;
	stressTestsRunning = 0;
	stressTimer = new QTimer();
	stressTimer->setSingleShot(true);
	connect(stressTimer, SIGNAL(timeout()), this, SLOT(stressTestEnded()));
	stressSubTimer = new QTimer();
	stressSubTimer->setSingleShot(false);
	stressSubTimer->setInterval(1000);
	stressTesting = false;
	connect(stressSubTimer, SIGNAL(timeout()), this, SLOT(stressTestProgress()));
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
			aWidget->setController(this);
			aWidget->setGpuName(QString(gpuString));
			aWidget->setGpuMemory(QString(gpuMemoryString));
			aWidget->setFont(QFont("Arial", 14));
			aWidget->setIndex(i);
			aWidget->setMaximumHeight(130);
			aWidget->setMinimumWidth(0);

			// Link up the test controller
			connect(aWidget, SIGNAL(testStarted(const int, bool)), this, SLOT(testStarted(const int, bool)));
			//connect(aWidget, SIGNAL(testEnded(const int)), this, SLOT(testEnded(const int)));

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
		if(deviceWidgets[i]->isChecked())
		{
			deviceWidgets[i]->startTestOnce();
		}
	}
}

void QtGpuMemtest::stopAll()
{
	for(int i = 0; i < deviceWidgets.count(); i++)
	{
		deviceWidgets[i]->endTest();
	}

	stressTestsRunning = 0;
}

//
// Aggregate test options
//

void QtGpuMemtest::quickTest()
{
	for(int i = 0; i < tests.count(); i++)
	{
		tests[i].testEnabled = (i < 6);
	}

	// current configuration for a quick test is tests 1-8 for each GPU, same as the template tests
	for(int i = 0; i < deviceWidgets.count(); i++)
	{
		deviceWidgets[i]->setTests(tests);
		deviceWidgets[i]->startTestOnce();
	}
}

void QtGpuMemtest::stressTest()
{
	stressTesting = true;

	for(int i = 0; i < tests.count(); i++)
	{
		tests[i].testEnabled = (tests[i].testName == tr("Test10 [Stress test]"));
	}

	// set all tests to stress test then go infinite and set a timer
	for(int i = 0; i < deviceWidgets.count(); i++)
	{
		deviceWidgets[i]->setTests(tests);
		deviceWidgets[i]->startTestInfinite();
		stressTestsRunning++;
	}

	// start the timer and configure the progress bar
	stressTimer->setInterval(ui.customStressDial->value() * 60 * 1000);
	stressTimer->start();
	ui.progressBarOverall->reset();
	ui.progressBarOverall->setMaximum(ui.customStressDial->value() * 60 + 5);
	ui.progressBarOverall->setValue(0);
	stressSubTimer->start();
}

//
// Test controller
//

void QtGpuMemtest::testsStarted(int n)
{
	if(stressTestsRunning == 0)	// stress test has it's own special progress bar handlers
	{
		// Reset and configure the progress bar
		if(allTestsDone)
		{
			ui.progressBarOverall->setMaximum(0);
			ui.progressBarOverall->setValue(0);
		}

		// Disable basic view controls
		ui.customStressTestButton->setEnabled(false);
		ui.customStressDial->setEnabled(false);
		ui.quickTestButton->setEnabled(false);

		ui.progressBarOverall->setMaximum(ui.progressBarOverall->maximum() + n);
		allTestsDone = false;
	}
}

void QtGpuMemtest::testEnded(const int index, QString testName)
{
	// Update progress bar progress, if all units done then go to results
	if(stressTestsRunning == 0) // again, defer stress test progress handling
	{
		ui.progressBarOverall->setValue(ui.progressBarOverall->value() + 1);
	}

	if((stressTesting && stressTestsRunning == 0) || (ui.progressBarOverall->value() == ui.progressBarOverall->maximum()))
	{
		stressTesting = false;
		stressTimer->stop();
		stressSubTimer->stop();

		// Reset progress bar, go to results
		ui.progressBarOverall->reset();
		allTestsDone = true;

		// Reset disabled basic view controls
		ui.customStressTestButton->setEnabled(true);
		ui.customStressDial->setEnabled(true);
		ui.quickTestButton->setEnabled(true);

		// TODO: go to results
		if(currentViewMode == BasicView)
		{
			// Setup results view
			bool aTestFailed = false;
			for(int i = 0; i < deviceWidgets.count(); i++)
			{
				if(deviceWidgets[i]->isTestFailed())
					aTestFailed = true;
			}

			if(aTestFailed)
			{
				ui.labelPassFail->setText(tr("One or more GPUs failed to pass the memory tests."));
			}
			else
			{
				ui.labelPassFail->setText(tr("All GPUs are OK! Passed the memory test."));
			}

			setView(BasicResultsView);
			// AdvancedResultsView is the same as AdvancedView lol
		}
	}
}

void QtGpuMemtest::stressTestProgress()
{
	ui.progressBarOverall->setValue(ui.progressBarOverall->value() + 1);
}

void QtGpuMemtest::stressTestEnded()
{
	// Stop the stress tests
	for(int i = 0; i < deviceWidgets.count(); i++)
	{
		deviceWidgets[i]->endTest();
		stressTestsRunning--;
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
// View Management
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

	for(int i = 0; i < deviceWidgets.count(); i++)
	{
		GpuDisplayWidget* thisWidget = deviceWidgets[i];
		sout << ">>> " << thisWidget->getName() << "\r\n\r\n" << thisWidget->getLog() << "\r\n\r\n";
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

	for(int i = 0; i < deviceWidgets.count(); i++)
	{
		GpuDisplayWidget* thisWidget = deviceWidgets[i];
		sout << ">>> " << thisWidget->getName() << "\r\n\r\n" << thisWidget->getLog() << "\r\n\r\n";
	}

	fout.close();
}

void QtGpuMemtest::handleBlockingError(int deviceIdx, int err, int cudaErr, QString line, QString file)
{
	QString errMessage;
	QTextStream(&errMessage) << "From line " << line << " in file " << file << ":\n";

	if(err == 0) QTextStream(&errMessage) << "General error code " << err << ".";
	else QTextStream(&errMessage) << "CUDA error code " << cudaErr << ".";

	QMessageBox::critical(this, "Error", errMessage);
	testThreads.remove(deviceIdx);
}

void QtGpuMemtest::handleNonBlockingError(int deviceIdx, int warn, QString line, QString file)
{
}

void QtGpuMemtest::handleProgress(int deviceIdx, int testNo, int action)
{
	QString progressMessage;
	QTextStream(&progressMessage) << deviceIdx << ": " << testNo << " action " << action;
	QMessageBox::information(this, "Progress Notification", progressMessage);
}