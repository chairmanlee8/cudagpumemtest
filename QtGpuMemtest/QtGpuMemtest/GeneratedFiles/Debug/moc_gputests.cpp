/****************************************************************************
** Meta object code from reading C++ file 'gputests.h'
**
** Created: Tue May 3 15:01:33 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../gputests.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'gputests.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_QtGpuThread[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       5,       // signalCount

 // signals: signature, parameters, type, tag, flags
      32,   13,   12,   12, 0x05,
      52,   13,   12,   12, 0x05,
      72,   13,   12,   12, 0x05,
      91,   13,   12,   12, 0x05,
     143,  113,   12,   12, 0x05,

 // slots: signature, parameters, type, tag, flags
     172,  168,   12,   12, 0x0a,
     187,   12,   12,   12, 0x0a,
     202,  200,   12,   12, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_QtGpuThread[] = {
    "QtGpuThread\0\0deviceIdx,testName\0"
    "failed(int,QString)\0passed(int,QString)\0"
    "ended(int,QString)\0starting(int,QString)\0"
    "deviceIdx,testName,logMessage\0"
    "log(int,QString,QString)\0idx\0"
    "setDevice(int)\0notifyExit()\0b\0"
    "setEndless(bool)\0"
};

const QMetaObject QtGpuThread::staticMetaObject = {
    { &QThread::staticMetaObject, qt_meta_stringdata_QtGpuThread,
      qt_meta_data_QtGpuThread, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &QtGpuThread::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *QtGpuThread::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *QtGpuThread::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_QtGpuThread))
        return static_cast<void*>(const_cast< QtGpuThread*>(this));
    return QThread::qt_metacast(_clname);
}

int QtGpuThread::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: failed((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 1: passed((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 2: ended((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 3: starting((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 4: log((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3]))); break;
        case 5: setDevice((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: notifyExit(); break;
        case 7: setEndless((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void QtGpuThread::failed(int _t1, QString _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void QtGpuThread::passed(int _t1, QString _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void QtGpuThread::ended(int _t1, QString _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void QtGpuThread::starting(int _t1, QString _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void QtGpuThread::log(int _t1, QString _t2, QString _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}
QT_END_MOC_NAMESPACE
