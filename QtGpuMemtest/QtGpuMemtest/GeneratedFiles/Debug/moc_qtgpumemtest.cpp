/****************************************************************************
** Meta object code from reading C++ file 'qtgpumemtest.h'
**
** Created: Tue May 3 15:50:27 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../qtgpumemtest.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'qtgpumemtest.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_QtGpuMemtest[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
      22,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      22,   14,   13,   13, 0x0a,
      54,   45,   13,   13, 0x0a,
      72,   13,   13,   13, 0x0a,
      80,   13,   13,   13, 0x0a,
      87,   13,   13,   13, 0x0a,
     101,   13,   13,   13, 0x0a,
     117,   13,   13,   13, 0x0a,
     132,   13,   13,   13, 0x0a,
     156,  148,   13,   13, 0x0a,
     177,   13,   13,   13, 0x0a,
     190,   13,   13,   13, 0x0a,
     205,   13,   13,   13, 0x0a,
     215,   13,   13,   13, 0x0a,
     230,  228,   13,   13, 0x0a,
     263,  248,   13,   13, 0x0a,
     286,   13,   13,   13, 0x0a,
     304,   13,   13,   13, 0x0a,
     357,  325,   13,   13, 0x0a,
     431,  406,   13,   13, 0x0a,
     503,  479,   13,   13, 0x0a,
     531,   13,   13,   13, 0x0a,
     543,   13,   13,   13, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_QtGpuMemtest[] = {
    "QtGpuMemtest\0\0minutes\0customStressValue(int)\0"
    "viewMode\0setView(ViewMode)\0about()\0"
    "exit()\0copyResults()\0exportResults()\0"
    "clearDevices()\0relistDevices()\0checked\0"
    "checkAllDevices(int)\0switchView()\0"
    "startChecked()\0stopAll()\0returnHome()\0"
    "n\0testsStarted(int)\0index,testName\0"
    "testEnded(int,QString)\0stressTestEnded()\0"
    "stressTestProgress()\0"
    "deviceIdx,err,cudaErr,line,file\0"
    "handleBlockingError(int,int,int,QString,QString)\0"
    "deviceIdx,warn,line,file\0"
    "handleNonBlockingError(int,int,QString,QString)\0"
    "deviceIdx,testNo,action\0"
    "handleProgress(int,int,int)\0quickTest()\0"
    "stressTest()\0"
};

const QMetaObject QtGpuMemtest::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_QtGpuMemtest,
      qt_meta_data_QtGpuMemtest, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &QtGpuMemtest::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *QtGpuMemtest::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *QtGpuMemtest::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_QtGpuMemtest))
        return static_cast<void*>(const_cast< QtGpuMemtest*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int QtGpuMemtest::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: customStressValue((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: setView((*reinterpret_cast< ViewMode(*)>(_a[1]))); break;
        case 2: about(); break;
        case 3: exit(); break;
        case 4: copyResults(); break;
        case 5: exportResults(); break;
        case 6: clearDevices(); break;
        case 7: relistDevices(); break;
        case 8: checkAllDevices((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 9: switchView(); break;
        case 10: startChecked(); break;
        case 11: stopAll(); break;
        case 12: returnHome(); break;
        case 13: testsStarted((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 14: testEnded((*reinterpret_cast< const int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 15: stressTestEnded(); break;
        case 16: stressTestProgress(); break;
        case 17: handleBlockingError((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3])),(*reinterpret_cast< QString(*)>(_a[4])),(*reinterpret_cast< QString(*)>(_a[5]))); break;
        case 18: handleNonBlockingError((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3])),(*reinterpret_cast< QString(*)>(_a[4]))); break;
        case 19: handleProgress((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 20: quickTest(); break;
        case 21: stressTest(); break;
        default: ;
        }
        _id -= 22;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
