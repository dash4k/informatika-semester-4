<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>WarJa</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css" />
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 p-8">

  <div class="max-w-5xl mx-auto">
    <h1 class="text-2xl font-bold mb-4">Manajemen Pengguna</h1>
    <div class="w-full flex flex-row justify-end gap-1">
        <button onclick="refresh()" class="mb-4 px-4 py-1 bg-blue-600 hover:bg-blue-300 text-white rounded"><i class="fas fa-sync-alt"></i></button>
        <button onclick="showForm()" class="mb-4 px-4 py-1 bg-green-600 hover:bg-green-300 text-white rounded"><i class="fa-solid fa-plus"></i></button>
    </div>

    <!-- Add/Edit Form -->
    <div id="form-container" class="fixed inset-0 bg-black bg-opacity-30 backdrop-blur-sm z-50 flex items-center justify-center opacity-0 pointer-events-none transition-all duration-300 ease-out">
        <div class="bg-white p-6 rounded shadow max-w-lg w-full transform transition-all duration-300 ease-in-out">
            <form id="user-form" class="flex flex-col gap-2">
                <div class="mb-2 flex flex-row justify-center items-center gap-1">
                    <div class="w-1/2">
                        <label for="user_id" class="block mb-2 text-sm font-light text-gray-900">ID</label>
                        <input type="number" id="user_id" name="user_id" placeholder="xx08561xxx" class="border p-2 w-full rounded-md">
                    </div>
                    <div class="w-1/2">
                        <label for="role" class="block mb-2 text-sm font-light text-gray-900">Role</label>
                        <select id="role" name="role" class="border p-2 py-3 w-full rounded-md bg-white h-auto">
                        <option value="user">User</option>
                        <option value="dosen">Dosen</option>
                        <option value="admin">Admin</option>
                        </select>
                    </div>
                </div>
                <div class="mb-2 flex flex-row justify-center gap-1">
                    <div>
                        <label for="email" class="block mb-2 text-sm font-light text-gray-900">Email</label>
                        <input type="email" id="email" name="email" placeholder="x@student.unud.ac.id" class="border p-2 w-full rounded-md">
                    </div>
                    <div>
                        <label for="password" class="block mb-2 text-sm font-light text-gray-900">Password</label>
                        <input type="password" id="password" name="password" placeholder="********" class="border p-2 w-full rounded-md">
                    </div>
                </div>
                <div class="mb-2">
                    <label for="name" class="block mb-2 text-sm font-light text-gray-900">Name</label>
                    <input type="text" id="name" name="name" placeholder="Agus Kopling" class="border p-2 w-full rounded-md">
                </div>
                <div class="mb-2">
                    <label for="profile" class="block mb-2 text-sm font-light text-gray-900">Profile</label>
                    <input type="file" id="profile" name="profile" class="border p-2 w-full rounded-md">
                    <p class="mt-2 text-xs text-gray-500">SVG, PNG, or JPG (MAX. 16 MB).</p>
                </div>
                <div class="flex gap-2 justify-end">
                    <button type="button" onclick="hideForm()" class="px-4 py-2 bg-gray-300 hover:bg-gray-200 rounded">Cancel</button>
                    <button type="button" onclick="submitForm()" class="px-4 py-2 bg-green-600 hover:bg-green-300 text-white rounded">Save</button>
                </div>
            </form>
        </div>
        </div>

    <!-- Users Table -->
    <table class="w-full bg-white shadow rounded">
      <thead>
        <tr class="bg-gray-200 text-left">
          <th class="p-2">User ID</th>
          <th class="p-2">Email</th>
          <th class="p-2">Password</th>
          <th class="p-2">Name</th>
          <th class="p-2">Role</th>
          <th class="p-2 text-center">Profile</th>
          <th class="p-2 text-center">Actions</th>
        </tr>
      </thead>
      <tbody id="user-table" class="text-left"></tbody>
    </table>
  </div>

  <script>
    let editingId = null;

    function refresh() {
        const btn = event.currentTarget;
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        fetchUsers();
        setTimeout(() => {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-sync-alt"></i>';
        }, 500);
    }

    function fetchUsers() {
        fetch('read.php')
            .then(res => res.json())
            .then(data => {
            const tbody = document.getElementById('user-table');
            tbody.innerHTML = '';

            if (data.length > 0) {
                data.forEach(user => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td class="p-2">${user.user_id}</td>
                    <td class="p-2">${user.email}</td>
                    <td class="p-2">********</td>
                    <td class="p-2">${user.name}</td>
                    <td class="p-2">${user.role}</td>
                    <td class="p-2 flex justify-center">${user.profile ? `<img src="data:image/jpeg;base64,${user.profile}" class="w-12 h-12 rounded-full">` : 'No image'}</td>
                    <td class="p-2 text-center">
                    <button class="edit-btn bg-yellow-300 hover:bg-yellow-200 px-4 py-1 rounded">Edit</button>
                    <button class="delete-btn bg-red-600 hover:bg-red-300 text-white px-2 py-1 ml-2 rounded">Delete</button>
                    </td>
                `;

                // Attach events
                row.querySelector('.edit-btn').addEventListener('click', () => editUser(user));
                row.querySelector('.delete-btn').addEventListener('click', () => deleteUser(user.user_id));

                tbody.appendChild(row);
                });
            } else {
                // No data found
                const row = document.createElement('tr');
                row.innerHTML = `
                <td class="p-4 text-center text-gray-500" colspan="7">No data available</td>
                `;
                tbody.appendChild(row);
            }
        });
    }

    function showForm() {
        const modal = document.getElementById('form-container');
        modal.classList.remove('opacity-0', 'pointer-events-none');
        modal.classList.add('opacity-100');
        document.getElementById('user-form').reset();
        editingId = null;
        }

    function hideForm() {
        const modal = document.getElementById('form-container');
        modal.classList.remove('opacity-100');
        modal.classList.add('opacity-0', 'pointer-events-none');
    }

    function editUser(user) {
      showForm();
      editingId = user.user_id;
      document.getElementById('user_id').value = user.user_id;
      document.getElementById('email').value = user.email;
      document.getElementById('password').value = '';
      document.getElementById('name').value = user.name;
      document.getElementById('role').value = user.role;
    }

    function submitForm() {
      const form = document.getElementById('user-form');
      const formData = new FormData(form);
      if (editingId) {
        formData.append('edit', 'true');
      }

      fetch(editingId ? 'update.php' : 'create.php', {
        method: 'POST',
        body: formData
      })
      .then(res => res.json())
      .then(() => {
        hideForm();
        fetchUsers();
      });
    }

    function deleteUser(id) {
      if (!confirm("Delete this user?")) return;
      fetch('delete.php', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id })
      })
      .then(res => res.json())
      .then(() => fetchUsers());
    }

    fetchUsers();
  </script>
</body>
</html>
