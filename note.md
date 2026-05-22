### Get afs on local 

```
kinit -f clement.gregoire@CRI.EPITA.FR
sshfs -o reconnect clement.gregoire@ssh.cri.epita.fr:/afs/cri.epita.fr/user/c/cl/clement.gregoire/u/ afs
```

warning
The files will not be accessible after the Kerberos ticket expires (usually after a few days). If this happens, ask for a new ticket by running `kinit -f login@CRI.EPITA.FR` and unmount the afs folder by running `umount afs/`. You can then mount it again with the same sshfs command as previously.


### Access CRI GPU

```
ssh -X -l clement.gregoire -p 22001 gpgpu.image.lrde.iaas.epita.fr
kinit
aklog
```


### Doc
```
https://docs.forge.epita.fr/epita/environment/from-home/afs/
https://moodle.epita.fr/mod/page/view.php?id=84549
```
